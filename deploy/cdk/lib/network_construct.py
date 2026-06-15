"""Network primitives for the RouteIQ Gateway AWS substrate (proposal P0 doc 31 §8).

Owns the VPC, the three security groups (ALB / pod / VPC-endpoint), the five
interface VPC endpoints, and the S3 gateway endpoint that RouteIQ's single
stateless gateway pod needs to boot and serve Bedrock from a private subnet.

This is a PORT of vllm-sr-on-aws/cdk/lib/network_construct.py, trimmed to the
RouteIQ-minimal set (proposal §8 / p0-discovery/discover-network.md §5):

- VSR's CIDR 10.20.0.0/16 -> RouteIQ 10.40.0.0/16 (distinct from VSR's
  10.20 ECS / 10.30 EKS so a future peering stays overlap-free).
- VSR's seven SGs -> three. The Envoy/dashboard ports (8080/8700) collapse to
  RouteIQ's single uvicorn port (4000). The efs / redis / milvus / backend SGs
  are DROPPED (RouteIQ is stateless, emptyDir only, no self-hosted backends);
  an aurora_sg is a P1 seam, not P0.
- VSR's eleven interface endpoints -> five. STS is DROPPED on the Pod Identity
  path (it was load-bearing only for IRSA's AssumeRoleWithWebIdentity; Pod
  Identity creds come from the pod-identity agent, not an STS endpoint call).
  SagemakerRuntime and the sr.internal Cloud Map namespace are DROPPED
  (VSR-specific). AppConfig x2 / aps-workspaces / XRay arrive with P2.
- VPC flow logs (ALL) are ADDED to satisfy AwsSolutions-VPC7 (proposal §8.5);
  the VSR source did not emit them.

All public attributes named here are the frozen interface contract for the
RouteIqStack composition root and the parallel CDK construct builders.
"""

from __future__ import annotations

from aws_cdk import aws_ec2 as ec2
from constructs import Construct

# Per mulch ec2-sg-description-charset, every description here must stay within
# the EC2 allowlist (a-zA-Z0-9 . _ - : / ( ) # , @ [ ] + = & ; { } ! $ *). No
# arrows, em-dashes, backticks, pipes, angle brackets, or question marks. An
# out-of-charset description passes cdk synth but FAILS the EC2 CREATE API.
_SG_DESCRIPTIONS: dict[str, str] = {
    "alb": "ALB ingress 443 from internet (or org allowlist via context)",
    "pod": "RouteIQ gateway pod ENIs (ingress from ALB only)",
    "vpce": "Shared SG for interface VPC endpoints",
}


class NetworkConstruct(Construct):
    """VPC, security groups, and VPC endpoints for the RouteIQ gateway pod."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        vpc_cidr: str = "10.40.0.0/16",
        nat_gateways: int = 1,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        self.vpc = ec2.Vpc(
            self,
            "Vpc",
            ip_addresses=ec2.IpAddresses.cidr(vpc_cidr),
            max_azs=2,
            nat_gateways=nat_gateways,
            enable_dns_hostnames=True,
            enable_dns_support=True,
            # VPC flow logs (ALL traffic) satisfy AwsSolutions-VPC7. CDK creates
            # the log group + the flow-log IAM role automatically when given a
            # default-keyed flow_logs entry (proposal §8.5).
            flow_logs={
                "AllTraffic": ec2.FlowLogOptions(
                    traffic_type=ec2.FlowLogTrafficType.ALL,
                ),
            },
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="private-app",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="private-data",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24,
                ),
            ],
        )

        self.public_subnets: list[ec2.ISubnet] = list(self.vpc.public_subnets)
        self.private_app_subnets: list[ec2.ISubnet] = list(
            self.vpc.select_subnets(subnet_group_name="private-app").subnets
        )
        self.private_data_subnets: list[ec2.ISubnet] = list(
            self.vpc.select_subnets(subnet_group_name="private-data").subnets
        )

        # Auto Mode managed load balancing discovers subnets by tag: public for
        # internet-facing LBs, private for internal LBs (proposal §8.2).
        self._tag_subnets_for_elb()

        # Per mulch cdk-nested-stack-sg-circular-dependency: instantiate every
        # SG BEFORE adding any ingress rule, so SG-to-SG references can be wired
        # without forcing a CFN dependency cycle (proposal §8.3).
        self.alb_sg = self._make_sg("AlbSg", "alb")
        self.pod_sg = self._make_sg("PodSg", "pod")
        self.vpce_sg = self._make_sg("VpceSg", "vpce")

        self._wire_sg_ingress()

        self.interface_endpoints: dict[str, ec2.InterfaceVpcEndpoint] = {}
        self._make_interface_endpoints()

        # Gateway (route-table) endpoint, not an ENI: no SG, free. Route-table
        # association covers BOTH private tiers so ECR layer blobs + config
        # download reach S3 from either subnet group (proposal §8.4).
        self.s3_gateway_endpoint = self.vpc.add_gateway_endpoint(
            "S3Gateway",
            service=ec2.GatewayVpcEndpointAwsService.S3,
            subnets=[
                ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
                ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED),
            ],
        )

    def _tag_subnets_for_elb(self) -> None:
        # kubernetes.io/role/elb=1 on public subnets, internal-elb=1 on the
        # private-app tier where the gateway pod ENIs live (proposal §8.2).
        from aws_cdk import Tags

        for subnet in self.public_subnets:
            Tags.of(subnet).add("kubernetes.io/role/elb", "1")
        for subnet in self.private_app_subnets:
            Tags.of(subnet).add("kubernetes.io/role/internal-elb", "1")

    def _make_sg(self, construct_id: str, key: str) -> ec2.SecurityGroup:
        return ec2.SecurityGroup(
            self,
            construct_id,
            vpc=self.vpc,
            description=_SG_DESCRIPTIONS[key],
            allow_all_outbound=True,
        )

    def _wire_sg_ingress(self) -> None:
        # alb_sg: 443 from world. PREP-ONLY at P0 (proposal §8.3 / §11.1a) -- the
        # chart default is ClusterIP / ingress.enabled=false, so the Auto-Mode
        # managed LB does not render and this SG backs nothing live until an
        # operator flips service.type to LoadBalancer. Narrowable to an org
        # allowlist via a future context flag.
        self.alb_sg.add_ingress_rule(
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port.tcp(443),
            description="HTTPS from internet to ALB",
        )

        # pod_sg: the gateway pod ENI SG. RouteIQ's container listens on a single
        # uvicorn port (default 4000), NOT VSR's Envoy 8080 / dashboard 8700.
        self.pod_sg.add_ingress_rule(
            peer=self.alb_sg,
            connection=ec2.Port.tcp(4000),
            description="Gateway uvicorn HTTP from ALB",
        )

        # vpce_sg: 443 from pod_sg only. This is the load-bearing one -- it gates
        # all five interface endpoints (proposal §8.3).
        self.vpce_sg.add_ingress_rule(
            peer=self.pod_sg,
            connection=ec2.Port.tcp(443),
            description="HTTPS from gateway pod to interface endpoints",
        )

    def _make_interface_endpoints(self) -> None:
        # The RouteIQ-minimal P0 set: five interface endpoints, all sharing
        # vpce_sg, all attached to the PRIVATE_WITH_EGRESS (private-app) subnets
        # where the gateway pod ENIs live, all private_dns_enabled (which the
        # VPC's enable_dns_hostnames + enable_dns_support make resolvable, or the
        # Bedrock/Secrets SDK would not resolve to the endpoint).
        #
        # DROPPED vs VSR (proposal §8.4 / discover-network §5): Sts (IRSA-only,
        # moot on Pod Identity), SagemakerRuntime (RouteIQ never calls SageMaker),
        # AppConfig / AppConfigData / Amp / XRay (P2). BEDROCK_RUNTIME requires
        # aws-cdk-lib >= 2.150.0.
        services: list[tuple[str, ec2.InterfaceVpcEndpointAwsService]] = [
            ("EcrApi", ec2.InterfaceVpcEndpointAwsService.ECR),
            ("EcrDocker", ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER),
            ("CloudWatchLogs", ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS),
            ("SecretsManager", ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER),
            ("BedrockRuntime", ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME),
        ]
        subnet_selection = ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)
        for construct_id, service in services:
            endpoint = self.vpc.add_interface_endpoint(
                construct_id,
                service=service,
                subnets=subnet_selection,
                security_groups=[self.vpce_sg],
                private_dns_enabled=True,
            )
            self.interface_endpoints[construct_id] = endpoint
