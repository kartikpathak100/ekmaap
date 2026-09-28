from .models import AuditLog


def log(db, actor, action: str, entity: str, entity_id=None, **detail):
    db.add(AuditLog(actor_id=getattr(actor, "id", None), action=action, entity=entity,
                    entity_id=str(entity_id) if entity_id else None, detail=detail or None))
