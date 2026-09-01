from types import ModuleType
from . import b2hhh
from . import delphes
from .common import Schema, detect_schema

_BACKENDS: dict[Schema, ModuleType] = {
    Schema.DELPHES: delphes,
    Schema.B2HHH: b2hhh,
    # Schema.DVNTUPLE: dvntuple,
}

def get_backend(
    schema: str | Schema | None = None, 
    *, 
    data = None
) -> ModuleType:
    if schema is None:
        if data is None:
            raise ValueError("Indicar 'schema' o 'data'.")
        selected_schema = detect_schema(data)
    elif isinstance(schema, Schema):
        selected_schema = schema
    else:
        try:
            selected_schema = Schema(str(schema).lower())
        except ValueError as e:
            available = ', '.join(item.value for item in Schema)
            raise ValueError(f"Esquema desconocido: {schema!r}.\nOpciones disponibles: {available}.") from e
    try:
        return _BACKENDS(selected_schema)
    except KeyError as e:
        implemented = ', '.join(item.value for item in _BACKENDS)
        raise NotImplementedError(f"No hay pipeline para {selected_schema.value!r}.\nImplementados: {implemented}.") from e