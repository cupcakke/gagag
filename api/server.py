from core.common import *
from api.routes import app, _create_fastapi_app, fastapi_app, configure_routes

def create_flask_app(runtime_instance: typing.Any, config: typing.Any) -> typing.Any:
    configure_routes(runtime_instance, config)
    return app

def create_fastapi_app(runtime_instance: typing.Any, config: typing.Any) -> typing.Any:
    configure_routes(runtime_instance, config)
    return fastapi_app
