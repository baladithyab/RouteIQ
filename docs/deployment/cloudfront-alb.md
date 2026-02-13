# CloudFront/ALB Settings for Streaming Responses

RouteIQ streaming responses (SSE for chat completions) require specific
CloudFront and ALB configuration. Without proper settings, users encounter
buffered responses, 504 timeouts, or connection drops mid-stream.

## ALB Settings

| Setting | Recommended Value | Why |
|---------|-------------------|-----|
| Idle timeout | 300s | Long streaming responses can take 2-3 min |
| Deregistration delay | 120s | Allow in-flight streams to complete during deploys |

## CloudFront Settings

| Setting | Recommended Value | Why |
|---------|-------------------|-----|
| Origin read timeout | 60s | Max wait for first byte from ALB |
| Cache policy | Disabled for `/v1/*` | API responses must not be cached |
| Origin request policy | AllViewer | Must forward Authorization header |
| Compress | Disabled for `/v1/*` | Compression breaks SSE streaming |
| Error caching | 10s for 502/503/504 | Fast recovery during deployments |

## Authorization Header (Important)

`Authorization` is a **restricted header** in CloudFront Origin Request Policies.
Using `OriginRequestHeaderBehavior.allowList("Authorization")` in CDK **silently
strips the header**. You must use `allViewer` to forward it:

```typescript
// WRONG: silently strips Authorization
const originPolicy = new cloudfront.OriginRequestPolicy(this, "Policy", {
  headerBehavior: cloudfront.OriginRequestHeaderBehavior.allowList("Authorization"),
});

// CORRECT: forwards all viewer headers including Authorization
const originPolicy = new cloudfront.OriginRequestPolicy(this, "Policy", {
  headerBehavior: cloudfront.OriginRequestHeaderBehavior.all(),
});
```

## Required SSE Response Headers

RouteIQ sets these automatically, but verify they pass through your proxy:

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
Transfer-Encoding: chunked
X-Accel-Buffering: no
```

## CDK Example

```typescript
const distribution = new cloudfront.Distribution(this, "CDN", {
  defaultBehavior: {
    origin: new origins.LoadBalancerV2Origin(alb, {
      protocolPolicy: cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
      readTimeout: Duration.seconds(60),
    }),
    viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
    cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
    originRequestPolicy: new cloudfront.OriginRequestPolicy(this, "AllViewer", {
      headerBehavior: cloudfront.OriginRequestHeaderBehavior.all(),
      queryStringBehavior: cloudfront.OriginRequestQueryStringBehavior.all(),
    }),
    compress: false,
  },
  errorResponses: [
    { httpStatus: 502, ttl: Duration.seconds(10) },
    { httpStatus: 503, ttl: Duration.seconds(10) },
    { httpStatus: 504, ttl: Duration.seconds(10) },
  ],
});
```
