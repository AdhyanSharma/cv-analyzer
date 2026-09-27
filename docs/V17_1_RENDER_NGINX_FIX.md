# V17.1 — Render Nginx Private-Network Fix

## Problem

The frontend deployment failed with:

```text
host not found in upstream "api"
```

The existing Nginx config was written for Docker Compose, where the API
service is reachable by the Compose DNS name:

```text
api:8000
```

Render deploys the frontend and API as separate services. They do not share
the Docker Compose DNS namespace, so `api` is not a valid Render hostname.

Render services in the same region can communicate through Render's private
network using internal service hostnames. Blueprint `fromService` references
can expose a service's `hostport` to another service.

## V17.1 fix

The frontend now uses:

```text
BACKEND_HOSTPORT
```

at runtime.

Render injects the API service's private host and port through:

```yaml
- key: BACKEND_HOSTPORT
  fromService:
    name: cv-analyzer-api
    type: web
    property: hostport
```

The Nginx official entrypoint then expands the template into the final
configuration.

For local Docker Compose, the custom entrypoint defaults to:

```text
api:8000
```

so the original local V15 behavior continues to work.

## Result

Local:

```text
React/Nginx
    |
    +--> api:8000
```

Render:

```text
Render Frontend
    |
    +--> Render private API hostname:8000
```

The browser continues using the same `/api/...` paths, so no frontend
source-code API URL rewrite is required.

## Files changed

```text
frontend/Dockerfile
frontend/nginx.conf.template
frontend/render-nginx-entrypoint.sh
render.yaml
```

The old `frontend/nginx.conf` can remain in the repository for reference,
but the new Dockerfile no longer copies it into the image.
