from types import ModuleType
from . import b2hhh
from . import delphes
from . import dvntuple

from ..common import (
    Schema,
    delta_phi,
    delta_r,
    detect_schema,
    energy,
    invariant_mass,
)

_BACKENDS: dict[Schema, ModuleType] = {
    Schema.DELPHES: delphes,
    Schema.DVNTUPLE: dvntuple,
    Schema.B2HHH: b2hhh,
}

def get_backend(
    schema: str | Schema | None = None,
    *,
    data=None
) -> ModuleType:
    if schema is None:
        if data is None:
            raise ValueError("Debes indicar 'schema' o proporcionar 'data'.")
        selected_schema = detect_schema(data)
    elif isinstance(schema, Schema):
        selected_schema = schema
    else:
        try:
            selected_schema = Schema(str(schema).lower())
        except ValueError as error:
            available = ", ".join(item.value for item in Schema)
            raise ValueError(f"Esquema desconocido: {schema!r}.\nOpciones disponibles: {available}.") from error
    return _BACKENDS[selected_schema]