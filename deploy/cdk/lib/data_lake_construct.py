"""DataLakeConstruct - routing_decision CloudWatch Logs to Firehose to S3
(Parquet) to Glue/Athena analytics lake. Flag-gated, off by default.

WHY this shape (P2 BUILD-NEW; research/p2/discover-datalake.md +
research/p2/build-outline.md FILE 3):
The RouteIQ gateway emits one structured ``routing_decision`` JSON line per
request to the dedicated CloudWatch Logs group P0 already created
(``/aws/containerinsights/<cluster>/routeiq-routing``). Those lines feed the
P2 metric filters, but there is no durable, query-able analytics lake - the
substrate for fine-tuning corpora, eval-set curation, offline routing-policy
evaluation, and per-tenant analytics.

The canonical AWS pipeline (no AMP, no app change beyond emitting the line):

    CW Logs SubscriptionFilter (filter "routing_decision")
      -> Kinesis Data Firehose delivery stream
          (CloudWatch-Logs decompression + JSON->Parquet via a Glue table schema)
      -> S3 ``routing-decisions/ingest_date=YYYY-MM-DD/`` (SNAPPY Parquet)
      -> Glue database + EXTERNAL table (partition-projected by ingest_date)
      -> Athena SQL / QuickSight

This is a near-verbatim port of vllm-sr-on-aws/cdk/lib/data_lake_construct.py,
re-derived symbol-by-symbol from the real source, with the RouteIQ divergences:

- Names re-based ``vllm_sr``/``vllm-sr`` -> ``routeiq``.
- ``source_log_group`` is the IMPORTED P0 routing log group (passed in by the
  composition root via ``logs.LogGroup.from_log_group_name`` - NOT a second
  group, NOT ``from_lookup``, to stay credential-free at synth).
- Every IAM role Description is ASCII / Latin-1 only (P0 lesson, proposal
  section 4.5): an em-dash (U+2014) passes ``cdk synth`` but FAILS the IAM
  CREATE API. The VSR source descriptions used em-dashes; this port uses
  plain hyphens.

SOURCE-LOG-GROUP / FLUENT-BIT CAVEAT (carried from the VSR source, lines
253-263, and recorded as a follow-up seed S2):
This construct expects the router's ``routing_decision`` JSON at the TOP LEVEL
of each application-log event. The source is P0's DEDICATED
``/aws/containerinsights/<cluster>/routeiq-routing`` group, where a Fluent Bit
pipeline must have JSON-parsed the router's stdout and promoted its keys to the
record top level. If the default Container Insights Fluent Bit forwards
container stdout WRAPPED (router JSON as a stringified ``log`` field +
``kubernetes.*`` metadata), the top-level event/selected_model/latency_ms
extraction yields nulls and a CloudWatchLogProcessing/Lambda transform to
unwrap ``log`` is needed. That Fluent Bit JSON-promotion is an app/chart-deploy
concern (a ``routeiq-routing.conf`` ``[INPUT]`` globbing the gateway pod + a
JSON parser), NOT a CDK deliverable; without it the lake + the dimensioned
metric filter see nulls.

DUAL-KEY NOTE (build-outline FILE 3 / D3): the lake's ``selected_model`` column
is the identity-SerDe column for the model actually used. The CW MetricFilter
(authored in observability_construct.py) keys its dimension on the OTel
telemetry-contract key ``$.["gen_ai.response.model"]`` (telemetry_contracts.py
:673 ``RESPONSE_MODEL``). The app emitter (router_decision_callback.py) emits
BOTH top-level keys with the SAME value: ``selected_model`` (this lake's
identity column) and ``gen_ai.response.model`` (the CW dimension key). This lake
schema does NOT carry the dotted ``gen_ai.response.model`` as a column - it is
carried in the line only for the metric-filter dimension; OpenXJsonSerDe's
``convert_dots_in_json_keys_to_underscores`` would otherwise fold it to
``gen_ai_response_model`` which is not in ``_COLUMNS`` and is harmlessly dropped.

PII posture: ``routing_decision`` lines carry model names, numeric scores, and
booleans only - NO prompt/completion text - so the lake needs no redaction.

Flag-gated: instantiated only when ``routeiq:enable_data_lake=true`` (wired by
the RouteIqObservabilityStack composition root), so the default synth is
byte-identical.

Public attributes: ``delivery_stream``, ``glue_database``, ``glue_table``,
``bucket`` (the lake bucket, caller-provided or own), ``kms_key``,
``subscription_filter``, ``lake_prefix``.
"""

from __future__ import annotations

from aws_cdk import Aws, Duration, RemovalPolicy
from aws_cdk import aws_glue as glue
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kinesisfirehose as firehose
from aws_cdk import aws_kms as kms
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from cdk_nag import NagSuppressions
from constructs import Construct

# The routing_decision line's columns - the flat-scalar field contract that the
# app-side emitter (router_decision_callback.py ``_emit_routing_decision_log``)
# writes at the TOP LEVEL of the JSON line, so OpenXJsonSerDe extracts them by
# identity name (no column-to-json-key mapping). This is the single most
# important artifact: it defines exactly what the structured log line must
# contain. 14 keys, ported verbatim from the VSR source (lines 69-84).
#
# NOTE (parity): there is deliberately NO ``cost`` / ``spend`` column and NO
# ``strategy`` column in the v1 lake schema - the lake captures tokens
# (prompt_tokens, completion_tokens) + latency_ms, not USD cost, and strategy
# lives on the span (router.strategy / gen_ai.routeiq.strategy), not the flat
# line. If per-tenant cost analytics or strategy-attribution is wanted, add a
# SECOND subscription+table keyed on a new line (mirroring the VSR
# ``routing_decision_bandit`` second-table pattern) - tracked as a follow-up
# seed (S5), not faked here.
_COLUMNS: list[tuple[str, str]] = [
    ("event", "string"),
    ("timestamp", "string"),
    ("request_id", "string"),
    ("trace_id", "string"),
    ("selected_model", "string"),
    ("model", "string"),
    ("decision", "string"),
    ("reason_code", "string"),
    ("category", "string"),
    ("reasoning_enabled", "boolean"),
    ("latency_ms", "int"),
    ("prompt_tokens", "int"),
    ("completion_tokens", "int"),
    ("cache_hit", "boolean"),
]

# The subscription filter selects the scalar routing_decision line by its
# top-level ``event`` key (JSON metric-filter form, NOT a bare substring). This
# requires the log line to be a JSON object with a top-level ``event`` key whose
# value is exactly the string ``"routing_decision"`` (underscore). It also
# excludes any sibling lines (e.g. a future ``routing_decision_bandit``).
_SUBSCRIPTION_FILTER_PATTERN = '{ $.event = "routing_decision" }'

# The Glue table name (also referenced by the Firehose schema configuration).
_TABLE_NAME = "routing_decisions"


class DataLakeConstruct(Construct):
    """routing_decision -> Firehose -> S3 Parquet -> Glue/Athena (flag-gated)."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        source_log_group: logs.ILogGroup,
        kms_key: kms.IKey | None = None,
        bucket: s3.IBucket | None = None,
        lake_prefix: str = "routing-decisions/",
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.env_name = env_name
        self.lake_prefix = lake_prefix
        region = Aws.REGION
        account = Aws.ACCOUNT_ID
        _retain = env_name != "dev"
        _owns_bucket = bucket is None  # construct provisions its own when unset

        # KMS: reuse the caller's CMK, or own one. Key rotation on; RETAIN in
        # non-dev so a stack destroy never orphans the lake's key into
        # PendingDeletion (the documented ECR-pull footgun). ASCII description.
        self.kms_key = kms_key or kms.Key(
            self,
            "LakeKey",
            description=f"RouteIQ {env_name} data-lake CMK (routing-decisions)",
            enable_key_rotation=True,
            removal_policy=RemovalPolicy.RETAIN if _retain else RemovalPolicy.DESTROY,
        )
        kms_key = self.kms_key

        # S3: reuse the caller's bucket, or own one with the standard posture
        # (KMS-SSE, versioned, BPA, enforce_ssl) + a lifecycle to Glacier/expire
        # the raw Parquet (cost control; tune per retention policy).
        self.bucket = bucket or s3.Bucket(
            self,
            "LakeBucket",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=kms_key,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            versioned=True,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN if _retain else RemovalPolicy.DESTROY,
            auto_delete_objects=not _retain,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="routing-decisions-tiering",
                    prefix=lake_prefix,
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90),
                        )
                    ],
                    expiration=Duration.days(730),
                    abort_incomplete_multipart_upload_after=Duration.days(7),
                )
            ],
        )
        bucket = self.bucket

        # -- 1. Glue database + table (the Parquet schema source of truth) ----
        db_name = f"routeiq_{env_name}_lake"
        self.glue_database = glue.CfnDatabase(
            self,
            "GlueDatabase",
            catalog_id=account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(name=db_name),
        )
        self.glue_table = glue.CfnTable(
            self,
            "RoutingDecisionsTable",
            catalog_id=account,
            database_name=db_name,
            table_input=glue.CfnTable.TableInputProperty(
                name=_TABLE_NAME,
                table_type="EXTERNAL_TABLE",
                # partition projection by ingest_date -> no MSCK/crawler needed
                # (build-outline D6: NO crawler).
                parameters={
                    "classification": "parquet",
                    "parquet.compression": "SNAPPY",
                    "projection.enabled": "true",
                    "projection.ingest_date.type": "date",
                    "projection.ingest_date.format": "yyyy-MM-dd",
                    "projection.ingest_date.range": "2026-01-01,NOW",
                    "storage.location.template": (
                        f"s3://{bucket.bucket_name}/{lake_prefix}ingest_date=${{ingest_date}}/"
                    ),
                },
                partition_keys=[glue.CfnTable.ColumnProperty(name="ingest_date", type="string")],
                storage_descriptor=glue.CfnTable.StorageDescriptorProperty(
                    location=f"s3://{bucket.bucket_name}/{lake_prefix}",
                    input_format=("org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"),
                    output_format=(
                        "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
                    ),
                    serde_info=glue.CfnTable.SerdeInfoProperty(
                        serialization_library=(
                            "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
                        )
                    ),
                    columns=[glue.CfnTable.ColumnProperty(name=n, type=t) for n, t in _COLUMNS],
                ),
            ),
        )
        self.glue_table.add_dependency(self.glue_database)

        # -- 2. Firehose delivery role (S3 + KMS + Glue read for schema) ------
        firehose_role = iam.Role(
            self,
            "FirehoseRole",
            assumed_by=iam.ServicePrincipal("firehose.amazonaws.com"),
            description=f"RouteIQ {env_name} routing-decisions Firehose delivery",
        )
        bucket.grant_read_write(firehose_role, objects_key_pattern=f"{lake_prefix}*")
        kms_key.grant_encrypt_decrypt(firehose_role)
        # Glue GetTable* so Firehose can read the Parquet schema for conversion.
        firehose_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "glue:GetTable",
                    "glue:GetTableVersion",
                    "glue:GetTableVersions",
                ],
                resources=[
                    f"arn:{Aws.PARTITION}:glue:{region}:{account}:catalog",
                    f"arn:{Aws.PARTITION}:glue:{region}:{account}:database/{db_name}",
                    f"arn:{Aws.PARTITION}:glue:{region}:{account}:table/{db_name}/{_TABLE_NAME}",
                ],
            )
        )

        # -- 3. Firehose delivery stream (decompress CW Logs + JSON->Parquet) -
        self.delivery_stream = firehose.CfnDeliveryStream(
            self,
            "DeliveryStream",
            delivery_stream_name=f"routeiq-{env_name}-routing-decisions",
            delivery_stream_type="DirectPut",
            extended_s3_destination_configuration=firehose.CfnDeliveryStream.ExtendedS3DestinationConfigurationProperty(
                bucket_arn=bucket.bucket_arn,
                role_arn=firehose_role.role_arn,
                prefix=f"{lake_prefix}ingest_date=!{{timestamp:yyyy-MM-dd}}/",
                error_output_prefix=(
                    f"{lake_prefix}errors/"
                    "!{firehose:error-output-type}/ingest_date=!{timestamp:yyyy-MM-dd}/"
                ),
                buffering_hints=firehose.CfnDeliveryStream.BufferingHintsProperty(
                    interval_in_seconds=300, size_in_m_bs=128
                ),
                # Parquet handles its own (SNAPPY) compression, so the Firehose
                # level is UNCOMPRESSED.
                compression_format="UNCOMPRESSED",
                encryption_configuration=firehose.CfnDeliveryStream.EncryptionConfigurationProperty(
                    kms_encryption_config=firehose.CfnDeliveryStream.KMSEncryptionConfigProperty(
                        awskms_key_arn=kms_key.key_arn
                    )
                ),
                # Decompress the gzipped CW Logs subscription payload + unwrap the
                # logEvents envelope so the inner routing_decision JSON is what
                # gets converted to Parquet. See the module-docstring Fluent Bit
                # caveat: the inner record must be flat top-level JSON.
                processing_configuration=firehose.CfnDeliveryStream.ProcessingConfigurationProperty(
                    enabled=True,
                    processors=[
                        # 1. Decompress the gzipped CW Logs subscription payload.
                        firehose.CfnDeliveryStream.ProcessorProperty(
                            type="Decompression",
                            parameters=[
                                firehose.CfnDeliveryStream.ProcessorParameterProperty(
                                    parameter_name="CompressionFormat",
                                    parameter_value="GZIP",
                                )
                            ],
                        ),
                        # 2. Unwrap the logEvents envelope -> inner routing_decision JSON.
                        firehose.CfnDeliveryStream.ProcessorProperty(
                            type="CloudWatchLogProcessing",
                            parameters=[
                                firehose.CfnDeliveryStream.ProcessorParameterProperty(
                                    parameter_name="DataMessageExtraction",
                                    parameter_value="true",
                                )
                            ],
                        ),
                    ],
                ),
                # JSON -> Parquet using the Glue table schema.
                data_format_conversion_configuration=firehose.CfnDeliveryStream.DataFormatConversionConfigurationProperty(
                    enabled=True,
                    input_format_configuration=firehose.CfnDeliveryStream.InputFormatConfigurationProperty(
                        deserializer=firehose.CfnDeliveryStream.DeserializerProperty(
                            # All lake columns are top-level JSON keys with matching
                            # names -> identity SerDe (no column_to_json_key_mappings
                            # needed). convert_dots_in_json_keys_to_underscores folds
                            # the dotted gen_ai.response.model dimension key (carried
                            # in the line for the CW filter) to gen_ai_response_model,
                            # which is not in _COLUMNS and is harmlessly dropped.
                            open_x_json_ser_de=firehose.CfnDeliveryStream.OpenXJsonSerDeProperty(
                                convert_dots_in_json_keys_to_underscores=True
                            )
                        )
                    ),
                    output_format_configuration=firehose.CfnDeliveryStream.OutputFormatConfigurationProperty(
                        serializer=firehose.CfnDeliveryStream.SerializerProperty(
                            parquet_ser_de=firehose.CfnDeliveryStream.ParquetSerDeProperty(
                                compression="SNAPPY"
                            )
                        )
                    ),
                    schema_configuration=firehose.CfnDeliveryStream.SchemaConfigurationProperty(
                        catalog_id=account,
                        database_name=db_name,
                        table_name=_TABLE_NAME,
                        region=region,
                        role_arn=firehose_role.role_arn,
                        version_id="LATEST",
                    ),
                ),
            ),
        )
        self.delivery_stream.add_dependency(self.glue_table)

        # -- 4. CW Logs SubscriptionFilter (routing_decision -> Firehose) -----
        # Use the L1 CfnSubscriptionFilter (NOT the L2 logs_destinations.
        # FirehoseDestination): that L2 helper expects an L2 firehose.
        # DeliveryStream, but the L2 DeliveryStream does not expose the Parquet
        # DataFormatConversionConfiguration we need - that lives only on the L1
        # CfnDeliveryStream. So we hand-build the CWL->Firehose delivery role
        # (logs.<region>.amazonaws.com assume + firehose:PutRecord*) and the
        # CfnSubscriptionFilter ourselves (the documented low-level path).
        cwl_to_firehose_role = iam.Role(
            self,
            "CwlToFirehoseRole",
            assumed_by=iam.ServicePrincipal(f"logs.{region}.amazonaws.com"),
            description=f"RouteIQ {env_name} CW Logs to routing-decisions Firehose",
        )
        cwl_to_firehose_role.add_to_policy(
            iam.PolicyStatement(
                actions=["firehose:PutRecord", "firehose:PutRecordBatch"],
                resources=[self.delivery_stream.attr_arn],
            )
        )
        # The source is P0's DEDICATED routeiq-routing log group (imported by the
        # composition root via from_log_group_name, NOT created here, NOT
        # from_lookup). The Fluent Bit pipeline on that group must have promoted
        # the router's JSON to the record top level (see module docstring + seed
        # S2) so this top-level exact-match works and OpenXJsonSerDe extracts the
        # flat columns.
        self.subscription_filter = logs.CfnSubscriptionFilter(
            self,
            "RoutingDecisionSubscription",
            log_group_name=source_log_group.log_group_name,
            filter_pattern=_SUBSCRIPTION_FILTER_PATTERN,
            destination_arn=self.delivery_stream.attr_arn,
            role_arn=cwl_to_firehose_role.role_arn,
        )

        # -- 5. cdk-nag suppressions (INLINE, evidence-bearing) ---------------
        # IAM5 on the Firehose role: the wildcards are the CDK grant helpers'
        # idiomatic action sets (grant_read_write -> s3:GetObject*/List*/etc;
        # grant_encrypt_decrypt -> kms:GenerateDataKey*/ReEncrypt*) scoped to the
        # lake bucket prefix + the single lake CMK. Firehose genuinely needs the
        # object-level S3 verbs (multipart put) and the data-key verbs (SSE-KMS).
        NagSuppressions.add_resource_suppressions(
            firehose_role,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": (
                        "CDK grant_read_write / grant_encrypt_decrypt emit these "
                        "action wildcards (s3:GetObject*/List*/Abort*/DeleteObject*, "
                        "kms:GenerateDataKey*/ReEncrypt*). They are scoped to the "
                        "lake bucket's routing-decisions/* prefix and the single "
                        "lake CMK; Firehose requires the object-level multipart-put "
                        "verbs + SSE-KMS data-key verbs to deliver Parquet. "
                        "Owner: RouteIQ P2 data-lake."
                    ),
                }
            ],
            apply_to_children=True,
        )
        # KDF1: the stream's S3 destination IS KMS-encrypted at rest
        # (encryption_configuration -> KMSEncryptionConfig above); the records are
        # SSE-KMS in S3. KDF1 wants stream-level SSE which is redundant for a
        # DirectPut stream whose only sink is the KMS'd lake bucket.
        NagSuppressions.add_resource_suppressions(
            self.delivery_stream,
            [
                {
                    "id": "AwsSolutions-KDF1",
                    "reason": (
                        "Delivered records are encrypted at rest with the lake CMK "
                        "via the S3 destination's KMSEncryptionConfig; stream-level "
                        "SSE is redundant for this DirectPut to KMS'd-S3 pipeline. "
                        "Owner: RouteIQ P2 data-lake."
                    ),
                }
            ],
        )
        # S1 on the lake bucket: it holds derived analytics Parquet (no PII -
        # routing_decision lines carry model names + numeric scores only), is
        # KMS-SSE + BPA + enforce_ssl + versioned. Server access logging would
        # add a second bucket for low marginal value on a derived-data store; the
        # access record of interest (who queried) is Athena/CloudTrail, not S3
        # GETs. Only suppress when the bucket is construct-OWNED (caller-supplied
        # buckets carry their own posture).
        if _owns_bucket:
            NagSuppressions.add_resource_suppressions(
                self.bucket,
                [
                    {
                        "id": "AwsSolutions-S1",
                        "reason": (
                            "Derived-analytics Parquet store (no PII; KMS-SSE + BPA "
                            "+ enforce_ssl + versioned). Query-access audit is via "
                            "Athena/CloudTrail, not S3 server access logs; a second "
                            "log bucket is low-value for this derived store. "
                            "Owner: RouteIQ P2 data-lake."
                        ),
                    }
                ],
            )

    @property
    def athena_database(self) -> str:
        """The Glue/Athena database name (for runbook / QuickSight wiring)."""
        return f"routeiq_{self.env_name}_lake"
