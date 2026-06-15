"""Unit tests for DataLakeConstruct (build-outline FILE 6 / test_data_lake.py).

Asserts the routing_decision CloudWatch Logs -> Kinesis Firehose -> S3 (Parquet)
-> Glue/Athena pipeline the P2 data-lake provisions (FILE 3): the Glue EXTERNAL
table with partition projection (NO crawler), the 14 ``_COLUMNS`` identity-SerDe
schema, the Firehose Decompression + CloudWatchLogProcessing processors + the
OpenXJsonSerDe->ParquetSerDe(SNAPPY) JSON-to-Parquet conversion at
``VersionId=LATEST``, the ``{ $.event = "routing_decision" }`` subscription
filter on the IMPORTED P0 routing log group, the S3 lifecycle, and the
ASCII-only IAM role descriptions.

The construct is FLAG-GATED off by default at the composition root
(``RouteIqObservabilityStack`` only instantiates it when
``routeiq:enable_data_lake=true``), so the "default OFF" assertion is that a
stack which does NOT instantiate the construct emits zero Firehose/Glue/table
resources - exercised here via a bare ``Stack`` with no DataLakeConstruct.

Synthesised offline against the dummy env (account ``123456789012`` /
``us-west-2``), credential-free: the source log group is IMPORTED by NAME via
``logs.LogGroup.from_log_group_name`` (NOT ``from_lookup``), mirroring how the
obs stack re-imports the P0 ``RoutingLogGroup`` from a prop. No second log group
is created here.

Per the suite convention (cdk-resource-count-test-tripwire) property assertions
use ``find_resources`` / ``has_resource_properties`` / ``Match.object_like`` over
brittle full counts on shared resources.
"""

from __future__ import annotations

import re
from typing import Any

import aws_cdk as cdk
from aws_cdk import aws_logs as logs
from aws_cdk.assertions import Match, Template

from lib.data_lake_construct import _COLUMNS, DataLakeConstruct

# The dummy account/region the cred-free gate pins (mirrors tests/conftest.py).
DUMMY_ACCOUNT = "123456789012"
DUMMY_REGION = "us-west-2"

# The P0-shaped routing log group name the obs stack re-imports from a prop.
_ROUTING_LOG_GROUP_NAME = "/aws/containerinsights/routeiq-dev/routeiq-routing"

# The 14 flat lake column names (identity SerDe) the table + the app emitter
# share. Mirrors data_lake_construct._COLUMNS.
_COLUMN_NAMES = [name for name, _type in _COLUMNS]

# The exact JSON metric-filter form the subscription uses (top-level event key).
_SUBSCRIPTION_FILTER_PATTERN = '{ $.event = "routing_decision" }'

# IAM's allowed Description charset: ASCII control trio + printable ASCII +
# Latin-1 supplement. An em-dash (U+2014) is OUTSIDE this set (the guarded
# failure mode: it passes ``cdk synth`` but FAILS the IAM CREATE API).
_IAM_DESCRIPTION_CHARSET = re.compile("^[" + "\t\n\r" + " -~" + "¡-ÿ" + "]*$")


def _dummy_env() -> cdk.Environment:
    return cdk.Environment(account=DUMMY_ACCOUNT, region=DUMMY_REGION)


def _data_lake_template(**construct_kwargs: Any) -> Template:
    """Synthesise a minimal stack holding ONE DataLakeConstruct (offline).

    The source log group is imported BY NAME (cred-free; no ``from_lookup``),
    exactly as the RouteIqObservabilityStack re-imports the P0 RoutingLogGroup
    from a prop. Per-test overrides flow through ``construct_kwargs``.
    """
    app = cdk.App()
    stack = cdk.Stack(app, "DataLakeTestStack", env=_dummy_env())
    source_log_group = logs.LogGroup.from_log_group_name(
        stack, "ImportedRoutingLogGroup", _ROUTING_LOG_GROUP_NAME
    )
    DataLakeConstruct(
        stack,
        "DataLakeConstruct",
        env_name=construct_kwargs.pop("env_name", "dev"),
        source_log_group=source_log_group,
        **construct_kwargs,
    )
    return Template.from_stack(stack)


def _bare_stack_template() -> Template:
    """A stack with NO DataLakeConstruct - the flag-OFF composition-root shape."""
    app = cdk.App()
    stack = cdk.Stack(app, "NoDataLakeStack", env=_dummy_env())
    # Importing the log group alone must not emit any lake resource.
    logs.LogGroup.from_log_group_name(stack, "ImportedRoutingLogGroup", _ROUTING_LOG_GROUP_NAME)
    return Template.from_stack(stack)


def test_data_lake_off_by_default_emits_no_resources() -> None:
    """Flag-OFF (construct not instantiated) emits no Firehose/Glue resources.

    The composition root instantiates DataLakeConstruct only when
    ``routeiq:enable_data_lake=true`` (build-outline D7 / FILE 4), so the default
    surface must carry zero data-lake resources.
    """
    template = _bare_stack_template()
    template.resource_count_is("AWS::KinesisFirehose::DeliveryStream", 0)
    template.resource_count_is("AWS::Glue::Table", 0)
    template.resource_count_is("AWS::Glue::Database", 0)
    template.resource_count_is("AWS::Logs::SubscriptionFilter", 0)


def test_no_glue_crawler() -> None:
    """Partition projection means NO Glue crawler (build-outline D6).

    Even when the lake IS instantiated, there is no ``AWS::Glue::Crawler`` - the
    EXTERNAL table is partition-projected on ingest_date, so no MSCK/crawler.
    """
    template = _data_lake_template()
    template.resource_count_is("AWS::Glue::Crawler", 0)


def test_glue_table_external_with_partition_projection() -> None:
    """Glue EXTERNAL_TABLE with partition projection on ingest_date.

    The table is EXTERNAL_TABLE; its parameters enable partition projection on a
    single ``ingest_date`` date partition (yyyy-MM-dd) with the storage-location
    template, so Athena needs no crawler / MSCK REPAIR (build-outline D6).
    """
    template = _data_lake_template()
    template.has_resource_properties(
        "AWS::Glue::Table",
        {
            "TableInput": Match.object_like(
                {
                    "Name": "routing_decisions",
                    "TableType": "EXTERNAL_TABLE",
                    "Parameters": Match.object_like(
                        {
                            "classification": "parquet",
                            "parquet.compression": "SNAPPY",
                            "projection.enabled": "true",
                            "projection.ingest_date.type": "date",
                            "projection.ingest_date.format": "yyyy-MM-dd",
                            "projection.ingest_date.range": "2026-01-01,NOW",
                        }
                    ),
                }
            )
        },
    )
    # The storage.location.template carries the bucket name as an Fn::Join
    # intrinsic (CFN-generated bucket name), so assert the literal suffix
    # (prefix + ingest_date partition placeholder) appears in the joined parts.
    tables = template.find_resources("AWS::Glue::Table")
    params = next(iter(tables.values()))["Properties"]["TableInput"]["Parameters"]
    location_tmpl = params["storage.location.template"]
    joined = "".join(part for part in location_tmpl["Fn::Join"][1] if isinstance(part, str))
    assert "routing-decisions/ingest_date=${ingest_date}/" in joined, joined
    assert joined.startswith("s3://"), joined


def test_glue_table_partition_key_is_ingest_date() -> None:
    """The single partition key is ``ingest_date`` (string)."""
    template = _data_lake_template()
    tables = template.find_resources("AWS::Glue::Table")
    assert len(tables) == 1, tables
    table_input = next(iter(tables.values()))["Properties"]["TableInput"]
    part_keys = table_input.get("PartitionKeys", [])
    names = [p.get("Name") for p in part_keys]
    assert names == ["ingest_date"], f"expected single ingest_date partition key, got {names}"
    assert part_keys[0].get("Type") == "string", part_keys


def test_glue_table_columns_match_the_14_lake_columns() -> None:
    """The table's storage-descriptor columns == the 14 ``_COLUMNS`` identity keys.

    OpenXJsonSerDe extracts by identity name, so the table column set IS the
    structured-log-line contract. Assert exactly those 14 names in order (no
    cost, no strategy column - v1 parity).
    """
    template = _data_lake_template()
    tables = template.find_resources("AWS::Glue::Table")
    table_input = next(iter(tables.values()))["Properties"]["TableInput"]
    columns = table_input["StorageDescriptor"]["Columns"]
    names = [c["Name"] for c in columns]
    assert names == _COLUMN_NAMES, f"table columns drifted from _COLUMNS: {names}"
    assert len(names) == 14, names
    assert "cost" not in names and "strategy" not in names, names


def test_glue_table_parquet_serde() -> None:
    """The storage descriptor uses the ParquetHiveSerDe + Mapred Parquet I/O formats."""
    template = _data_lake_template()
    tables = template.find_resources("AWS::Glue::Table")
    sd = next(iter(tables.values()))["Properties"]["TableInput"]["StorageDescriptor"]
    assert "MapredParquetInputFormat" in sd["InputFormat"], sd["InputFormat"]
    assert "MapredParquetOutputFormat" in sd["OutputFormat"], sd["OutputFormat"]
    assert "ParquetHiveSerDe" in sd["SerdeInfo"]["SerializationLibrary"], sd["SerdeInfo"]


def test_glue_table_depends_on_database() -> None:
    """The Glue table DependsOn the Glue database (it cannot exist before it)."""
    template = _data_lake_template()
    res = template.to_json()["Resources"]
    db_ids = [lid for lid, r in res.items() if r["Type"] == "AWS::Glue::Database"]
    table_ids = [lid for lid, r in res.items() if r["Type"] == "AWS::Glue::Table"]
    assert len(db_ids) == 1 and len(table_ids) == 1, (db_ids, table_ids)
    depends_on = res[table_ids[0]].get("DependsOn") or []
    if isinstance(depends_on, str):
        depends_on = [depends_on]
    assert db_ids[0] in depends_on, (
        f"Glue table must DependsOn the Glue database ({db_ids[0]}); DependsOn={depends_on}"
    )


def test_firehose_stream_directput_with_named_stream() -> None:
    """The Firehose delivery stream is DirectPut and env-named."""
    template = _data_lake_template()
    template.has_resource_properties(
        "AWS::KinesisFirehose::DeliveryStream",
        {
            "DeliveryStreamName": "routeiq-dev-routing-decisions",
            "DeliveryStreamType": "DirectPut",
        },
    )


def test_firehose_processors_decompression_then_cwl_extraction() -> None:
    """Firehose runs Decompression(GZIP) then CloudWatchLogProcessing(extract).

    That processor pair unwraps the gzipped CW Logs subscription envelope so the
    inner routing_decision JSON is what gets converted to Parquet.
    """
    template = _data_lake_template()
    streams = template.find_resources("AWS::KinesisFirehose::DeliveryStream")
    dest = next(iter(streams.values()))["Properties"]["ExtendedS3DestinationConfiguration"]
    procs = dest["ProcessingConfiguration"]
    assert procs["Enabled"] is True, procs
    proc_types = [p["Type"] for p in procs["Processors"]]
    assert proc_types == ["Decompression", "CloudWatchLogProcessing"], proc_types

    # Decompression CompressionFormat=GZIP.
    decompress = procs["Processors"][0]
    decompress_params = {p["ParameterName"]: p["ParameterValue"] for p in decompress["Parameters"]}
    assert decompress_params.get("CompressionFormat") == "GZIP", decompress_params

    # CloudWatchLogProcessing DataMessageExtraction=true.
    cwl = procs["Processors"][1]
    cwl_params = {p["ParameterName"]: p["ParameterValue"] for p in cwl["Parameters"]}
    assert cwl_params.get("DataMessageExtraction") == "true", cwl_params


def test_firehose_json_to_parquet_conversion() -> None:
    """OpenXJsonSerDe -> ParquetSerDe(SNAPPY) conversion keyed to the table @ LATEST.

    Input deserializer is OpenXJsonSerDe (with dot-to-underscore folding on);
    output serializer is ParquetSerDe SNAPPY; the schema configuration points at
    the routing_decisions Glue table at VersionId=LATEST. Firehose-level
    compression is UNCOMPRESSED (Parquet does its own SNAPPY).
    """
    template = _data_lake_template()
    streams = template.find_resources("AWS::KinesisFirehose::DeliveryStream")
    dest = next(iter(streams.values()))["Properties"]["ExtendedS3DestinationConfiguration"]
    assert dest["CompressionFormat"] == "UNCOMPRESSED", dest["CompressionFormat"]

    conv = dest["DataFormatConversionConfiguration"]
    assert conv["Enabled"] is True, conv

    deser = conv["InputFormatConfiguration"]["Deserializer"]
    assert "OpenXJsonSerDe" in deser, deser
    assert deser["OpenXJsonSerDe"].get("ConvertDotsInJsonKeysToUnderscores") is True, deser

    ser = conv["OutputFormatConfiguration"]["Serializer"]
    assert "ParquetSerDe" in ser, ser
    assert ser["ParquetSerDe"].get("Compression") == "SNAPPY", ser

    schema = conv["SchemaConfiguration"]
    assert schema["TableName"] == "routing_decisions", schema
    assert schema["VersionId"] == "LATEST", schema


def test_firehose_buffering_and_kms_encryption() -> None:
    """Buffering 300s/128MB and the S3 destination is SSE-KMS (lake CMK)."""
    template = _data_lake_template()
    streams = template.find_resources("AWS::KinesisFirehose::DeliveryStream")
    dest = next(iter(streams.values()))["Properties"]["ExtendedS3DestinationConfiguration"]
    buffering = dest["BufferingHints"]
    assert buffering["IntervalInSeconds"] == 300, buffering
    assert buffering["SizeInMBs"] == 128, buffering
    # SSE-KMS via KMSEncryptionConfig (an Fn::GetAtt to the lake CMK).
    assert "KMSEncryptionConfig" in dest["EncryptionConfiguration"], dest["EncryptionConfiguration"]


def test_firehose_prefix_partitions_by_ingest_date() -> None:
    """The S3 prefix partitions by ingest_date=!{timestamp:yyyy-MM-dd}."""
    template = _data_lake_template()
    streams = template.find_resources("AWS::KinesisFirehose::DeliveryStream")
    dest = next(iter(streams.values()))["Properties"]["ExtendedS3DestinationConfiguration"]
    assert dest["Prefix"] == "routing-decisions/ingest_date=!{timestamp:yyyy-MM-dd}/", dest[
        "Prefix"
    ]
    assert "errors/" in dest["ErrorOutputPrefix"], dest["ErrorOutputPrefix"]


def test_firehose_stream_depends_on_glue_table() -> None:
    """The delivery stream DependsOn the Glue table (schema must exist first)."""
    template = _data_lake_template()
    res = template.to_json()["Resources"]
    table_ids = [lid for lid, r in res.items() if r["Type"] == "AWS::Glue::Table"]
    stream_ids = [
        lid for lid, r in res.items() if r["Type"] == "AWS::KinesisFirehose::DeliveryStream"
    ]
    assert len(table_ids) == 1 and len(stream_ids) == 1, (table_ids, stream_ids)
    depends_on = res[stream_ids[0]].get("DependsOn") or []
    if isinstance(depends_on, str):
        depends_on = [depends_on]
    assert table_ids[0] in depends_on, (
        f"delivery stream must DependsOn the Glue table ({table_ids[0]}); DependsOn={depends_on}"
    )


def test_subscription_filter_pattern_and_source_group() -> None:
    """The CW Logs subscription filter selects ``{ $.event = "routing_decision" }``.

    It reads the IMPORTED P0 routing log group by NAME and routes to the Firehose
    stream via the hand-built CWL->Firehose role.
    """
    template = _data_lake_template()
    template.has_resource_properties(
        "AWS::Logs::SubscriptionFilter",
        {
            "FilterPattern": _SUBSCRIPTION_FILTER_PATTERN,
            "LogGroupName": _ROUTING_LOG_GROUP_NAME,
        },
    )


def test_cwl_to_firehose_role_assumes_logs_service_principal() -> None:
    """The CWL->Firehose role is assumable by logs.<region>.amazonaws.com.

    With firehose:PutRecord* on the delivery stream ARN. (Region-qualified
    logs principal is required for a CW Logs subscription destination role.)
    """
    template = _data_lake_template()
    roles = template.find_resources("AWS::IAM::Role")
    logs_principal_roles = [
        r
        for r in roles.values()
        if "logs." in str(r["Properties"].get("AssumeRolePolicyDocument", {}))
    ]
    assert logs_principal_roles, "no role assumable by the logs service principal found"


def test_s3_lifecycle_glacier_90_expire_730() -> None:
    """Lake bucket lifecycle: Glacier @90d, expire @730d, abort multipart @7d."""
    template = _data_lake_template()
    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "LifecycleConfiguration": Match.object_like(
                {
                    "Rules": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "Transitions": Match.array_with(
                                        [
                                            Match.object_like(
                                                {
                                                    "StorageClass": "GLACIER",
                                                    "TransitionInDays": 90,
                                                }
                                            )
                                        ]
                                    ),
                                    "ExpirationInDays": 730,
                                    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7},
                                }
                            )
                        ]
                    )
                }
            )
        },
    )


def test_s3_bucket_kms_bpa_versioned_ssl() -> None:
    """Lake bucket is KMS-SSE + BlockPublicAccess(all) + versioned + enforce_ssl."""
    template = _data_lake_template()
    # KMS SSE + versioning on the bucket resource.
    template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "BucketEncryption": Match.object_like(
                {
                    "ServerSideEncryptionConfiguration": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "ServerSideEncryptionByDefault": Match.object_like(
                                        {"SSEAlgorithm": "aws:kms"}
                                    )
                                }
                            )
                        ]
                    )
                }
            ),
            "VersioningConfiguration": {"Status": "Enabled"},
            "PublicAccessBlockConfiguration": Match.object_like(
                {
                    "BlockPublicAcls": True,
                    "BlockPublicPolicy": True,
                    "IgnorePublicAcls": True,
                    "RestrictPublicBuckets": True,
                }
            ),
        },
    )
    # enforce_ssl emits a DenyInsecure bucket policy with aws:SecureTransport=false.
    template.has_resource_properties(
        "AWS::S3::BucketPolicy",
        {
            "PolicyDocument": Match.object_like(
                {
                    "Statement": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "Effect": "Deny",
                                    "Condition": {"Bool": {"aws:SecureTransport": "false"}},
                                }
                            )
                        ]
                    )
                }
            )
        },
    )


def test_own_kms_cmk_with_rotation() -> None:
    """When no CMK is supplied, the lake owns a CMK with key rotation enabled."""
    template = _data_lake_template()
    template.has_resource_properties(
        "AWS::KMS::Key",
        {"EnableKeyRotation": True},
    )


def test_caller_supplied_kms_and_bucket_are_reused() -> None:
    """Supplying kms_key + bucket reuses them: no own CMK, no own bucket, no S1.

    Exercises the ``_owns_bucket`` branch (S1 suppression only when owned) and the
    caller-CMK path so the construct does not double-provision.
    """
    from aws_cdk import aws_kms as kms
    from aws_cdk import aws_s3 as s3

    app = cdk.App()
    stack = cdk.Stack(app, "ReuseStack", env=_dummy_env())
    source_log_group = logs.LogGroup.from_log_group_name(
        stack, "ImportedRoutingLogGroup", _ROUTING_LOG_GROUP_NAME
    )
    supplied_key = kms.Key(stack, "SuppliedKey", enable_key_rotation=True)
    supplied_bucket = s3.Bucket(
        stack,
        "SuppliedBucket",
        encryption=s3.BucketEncryption.KMS,
        encryption_key=supplied_key,
        enforce_ssl=True,
    )
    DataLakeConstruct(
        stack,
        "DataLakeConstruct",
        env_name="dev",
        source_log_group=source_log_group,
        kms_key=supplied_key,
        bucket=supplied_bucket,
    )
    template = Template.from_stack(stack)
    # Exactly the supplied CMK + supplied bucket (no construct-owned LakeKey/LakeBucket).
    template.resource_count_is("AWS::KMS::Key", 1)
    template.resource_count_is("AWS::S3::Bucket", 1)


def test_iam_role_descriptions_are_ascii() -> None:
    """Every IAM role Description is ASCII / Latin-1 only (P0 lesson 4.5).

    An em-dash (U+2014) passes ``cdk synth`` but FAILS the IAM CREATE API. The
    VSR source descriptions used em-dashes; this port must use plain hyphens.
    """
    template = _data_lake_template()
    roles = template.find_resources("AWS::IAM::Role")
    for logical, role in roles.items():
        desc = role["Properties"].get("Description")
        if isinstance(desc, str):
            assert _IAM_DESCRIPTION_CHARSET.match(desc), (
                f"IAM role {logical} Description has a char outside IAM's allowed "
                f"Latin-1 set (e.g. an em-dash): {desc!r}"
            )


def test_athena_database_name() -> None:
    """The athena_database property is ``routeiq_<env>_lake``."""
    app = cdk.App()
    stack = cdk.Stack(app, "AthenaNameStack", env=_dummy_env())
    source_log_group = logs.LogGroup.from_log_group_name(
        stack, "ImportedRoutingLogGroup", _ROUTING_LOG_GROUP_NAME
    )
    construct = DataLakeConstruct(
        stack,
        "DataLakeConstruct",
        env_name="prod",
        source_log_group=source_log_group,
    )
    assert construct.athena_database == "routeiq_prod_lake"
