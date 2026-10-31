from fastapi import APIRouter, Header, HTTPException, Response, status

from app.core.metrics import metrics_enabled, metrics_token, render_metrics

router = APIRouter()


@router.get("/metrics")
def prometheus_metrics(authorization: str | None = Header(default=None)) -> Response:
    if not metrics_enabled():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    expected = metrics_token()
    if expected:
        provided = ""
        if authorization and authorization.lower().startswith("bearer "):
            provided = authorization.split(" ", 1)[1].strip()
        if provided != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)
