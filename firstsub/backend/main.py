try:
    from .app.main import app
except ImportError:  # pragma: no cover
    from app.main import app
