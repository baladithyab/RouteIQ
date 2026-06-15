# EKS Auto Mode: ALB Ingress via IngressClassParams (Runbook)

This runbook covers the **EKS Auto Mode** edge for RouteIQ Gateway. Auto Mode
inverts the familiar self-managed AWS Load Balancer Controller (LBC) model:
AWS-level ALB configuration moves **off** the `Ingress` object's
`alb.ingress.kubernetes.io/*` annotations and **into** an `IngressClassParams`
custom resource. Two RouteIQ-specific consequences fall out of that inversion,
and both are covered below.

> **When does this apply?** Only on clusters running **EKS Auto Mode** (the
> built-in, AWS-managed load-balancing capability). If you run the **self-managed
> AWS Load Balancer Controller**, the annotation path documented in the chart
> `values.yaml` (`ingress.annotations: alb.ingress.kubernetes.io/*`) is correct
> and you do **not** need this runbook. Both paths are supported; this document is
> the Auto Mode variant.

## TL;DR

1. ALB config (`certificateARNs`, explicit `targetType`, scheme) lives in an
   `IngressClassParams` CR, referenced by an `IngressClass`, referenced by the
   `Ingress`. It is **not** set via `alb.ingress.kubernetes.io/*` annotations on
   the Ingress.
2. **ALB-level OIDC authentication (`alb.ingress.kubernetes.io/auth-type: oidc`)
   is NOT supported on Auto Mode.** OIDC/SSO must terminate **in the gateway app**
   (`oidc.py`) — set `oidc.enabled=true` and wire the client secret (see the OIDC
   section in [Security](security.md)).

## Why Auto Mode is different

The self-managed LBC reads ALB settings from per-`Ingress`
`alb.ingress.kubernetes.io/*` annotations. Under Auto Mode, the managed
controller instead reads them from an `IngressClassParams` resource attached to
the `IngressClass` the `Ingress` selects. Annotations that the self-managed
controller honoured are either ignored or only partially honoured. The two that
matter most for RouteIQ:

| Setting | Self-managed LBC (annotation) | Auto Mode (IngressClassParams) |
|---------|-------------------------------|--------------------------------|
| ACM certificate | `alb.ingress.kubernetes.io/certificate-arn` (single ARN string) | `spec.certificateARNs` (a **list**) |
| Target type | `alb.ingress.kubernetes.io/target-type: ip` | `spec.targetType: ip` (set **explicitly**) |
| Scheme | `alb.ingress.kubernetes.io/scheme: internet-facing` | `spec.scheme: internet-facing` |
| ALB OIDC auth | `alb.ingress.kubernetes.io/auth-type: oidc` (+ `auth-idp-oidc`) | **Unsupported** — terminate OIDC in the gateway app |

Set `targetType` **explicitly** rather than relying on a default — RouteIQ pods
register as ENI targets (`ip`), and an implicit `instance` target type will
mis-route or fail health checks.

## Steps

### 1. Create the IngressClassParams CR

```yaml
apiVersion: eks.amazonaws.com/v1
kind: IngressClassParams
metadata:
  name: routeiq-alb
spec:
  scheme: internet-facing
  # Explicit ip target type: RouteIQ pods register as ENI (ip) targets.
  targetType: ip
  # ACM cert ARNs as a LIST (not the single-string annotation form).
  # Issue the cert out-of-band; <region>/<acct>/<id> are operator values.
  certificateARNs:
    - arn:aws:acm:<region>:<acct>:certificate/<id>
  # Optional: pin listener ports / SSL policy here if your Auto Mode version
  # supports it; otherwise listener config is inferred from the Ingress + certs.
```

### 2. Create an IngressClass that references the params

```yaml
apiVersion: networking.k8s.io/v1
kind: IngressClass
metadata:
  name: routeiq-alb
spec:
  controller: eks.amazonaws.com/alb
  parameters:
    apiGroup: eks.amazonaws.com
    kind: IngressClassParams
    name: routeiq-alb
```

### 3. Point the chart Ingress at that IngressClass

In `values.yaml` (or via `--set` at `helm upgrade`):

```yaml
ingress:
  enabled: true
  className: routeiq-alb          # the IngressClass created in step 2
  # Under Auto Mode the ALB config is in IngressClassParams, so the
  # alb.ingress.kubernetes.io/* annotations are NOT used here. Leave annotations
  # empty (or limit them to non-ALB-config annotations).
  annotations: {}
  hosts:
    - host: routeiq.<domain>
      paths:
        - path: /
          pathType: Prefix
  # Auto Mode terminates TLS at the ALB using the certificateARNs from
  # IngressClassParams, so the chart-level tls[] (K8s Secret-backed TLS) is
  # typically left empty on this path.
  tls: []
```

The chart's `templates/ingress.yaml` emits `spec.ingressClassName` from
`ingress.className` — that is the only wiring the chart needs; everything ALB
moves to the CR.

### 4. Terminate OIDC/SSO in the gateway, not at the ALB

ALB-level OIDC auth (`auth-type: oidc`) is unsupported on Auto Mode, so the ALB
forwards requests unauthenticated and the **gateway app** performs the OIDC
exchange. Enable it via the chart:

```yaml
oidc:
  enabled: true
  issuerUrl: https://your-tenant.example.com/
  clientId: routeiq-gateway
  existingSecret: <release>-routeiq-gateway-secrets   # confidential-client secret
  existingSecretKey: oidc-client-secret
```

- The gateway exposes `/sso/login`, `/sso/callback`, and `/auth/token-exchange`
  (see `oidc.py`). The **`/sso/callback` URL must be public** (reachable through
  the ALB edge above) and **registered with the IdP out-of-band** as a redirect
  URI.
- If `oidc.existingSecret` is left unset, the gateway silently degrades to OIDC
  **public-client mode** (no client authentication). The Helm `NOTES` output
  warns loudly on this path. See the OIDC section in [Security](security.md) for
  the confidential-client wiring and the public-client fallback caveat.

> **Multi-pod note.** With more than one replica, minted `sk-oidc-*` keys and the
> identity cache are in-process only today (sticky-session dependent). A
> Redis-backed shared store is deferred. If you front the gateway with the ALB and
> run multiple replicas, enable session stickiness at the ALB until the shared
> store ships.

## Verification

1. `kubectl get ingressclassparams routeiq-alb -o yaml` — confirm
   `certificateARNs`, `targetType: ip`, and `scheme` are populated.
2. `kubectl get ingress <release>-routeiq-gateway -o jsonpath='{.spec.ingressClassName}'`
   returns `routeiq-alb`.
3. `kubectl describe ingress <release>-routeiq-gateway` — confirm the managed
   controller provisioned an ALB and the address resolves.
4. Hit the public host over HTTPS and confirm the ACM cert terminates at the ALB:
   `curl -sv https://routeiq.<domain>/_health/live`.
5. Drive the SSO flow end to end (`/sso/login` → IdP → `/sso/callback`) to confirm
   the gateway-terminated OIDC exchange works and the IdP accepts the registered
   callback URL.

## Common failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| ALB provisions but serves the default cert / TLS handshake fails | `certificate-arn` set as an annotation instead of `certificateARNs` in IngressClassParams | Move the ARN(s) into `IngressClassParams.spec.certificateARNs` (a list) |
| Targets unhealthy / 504s | Implicit `instance` target type | Set `spec.targetType: ip` explicitly |
| SSO redirect loop or `redirect_uri` rejected | `/sso/callback` not registered with the IdP, or not publicly reachable | Register the public callback URL at the IdP; confirm `ingress.enabled=true` and the ALB host resolves |
| SSO works on one request, fails the next | Multi-pod without sticky sessions (in-process key/identity cache) | Enable ALB session stickiness until the Redis-backed store ships |
| `auth-type: oidc` annotation appears to do nothing | ALB-level OIDC is unsupported on Auto Mode | Terminate OIDC in the gateway (`oidc.enabled=true`) instead |
