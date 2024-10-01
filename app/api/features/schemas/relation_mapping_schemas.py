from typing import List, Optional
from pydantic import BaseModel, Field, conlist

class RelationMappingInputData(BaseModel):
    topic: str = Field(..., description="The main topic")
    file_url: str = Field(..., description="URL of the file to be processed")
    file_type: str = Field(..., description="Type of the file (e.g., csv, json, xls, xlsx, xml)")
    lang: str = Field(..., description="Language of the document")

class EntityAttribute(BaseModel):
    attribute_name: str = Field(..., description="Name of the entity's attribute")
    attribute_type: str = Field(..., description="Type of the attribute (e.g., 'string', 'integer', etc.)")
    is_primary_key: bool = Field(False, description="Indicates if this attribute is a primary key")
    is_foreign_key: bool = Field(False, description="Indicates if this attribute is a foreign key")
    constraints: Optional[List[str]] = Field(None, description="List of constraints on the attribute (e.g., 'NOT NULL', 'UNIQUE')")

class Entity(BaseModel):
    entity_id: str = Field(..., description="Unique identifier for the entity")
    entity_name: str = Field(..., description="Human-readable name of the entity")
    attributes: List[EntityAttribute] = Field(..., description="List of attributes defining the entity")
    description: Optional[str] = Field(None, description="Optional description of the entity")
    entity_type: Optional[str] = Field(None, description="Type of the entity (e.g., 'table', 'class', etc.)")

class RelationCardinality(BaseModel):
    source_cardinality: str = Field(..., description="Cardinality of the source entity in the relation (e.g., 'one', 'many')")
    target_cardinality: str = Field(..., description="Cardinality of the target entity in the relation (e.g., 'one', 'many')")
    description: Optional[str] = Field(None, description="Optional description of the cardinality")

class RelationConstraint(BaseModel):
    constraint_type: str = Field(..., description="Type of constraint on the relation (e.g., 'cascade delete', 'restrict')")
    applies_to: str = Field(..., description="Specifies whether the constraint applies to the source or target entity")
    description: Optional[str] = Field(None, description="Optional description of the constraint")

class RelationWeight(BaseModel):
    weight_value: float = Field(..., description="The weight or strength of the relation")
    weight_type: Optional[str] = Field(None, description="Type of the weight (e.g., 'importance', 'confidence')")
    description: Optional[str] = Field(None, description="Optional description of what the weight represents")

class Relation(BaseModel):
    relation_id: str = Field(..., description="Unique identifier for the relation")
    source_entity: str = Field(..., description="ID of the entity from which the relation originates")
    target_entity: str = Field(..., description="ID of the entity to which the relation points")
    relation_type: str = Field(..., description="Type of relation (e.g., 'one-to-many', 'many-to-many', 'inheritance')")
    cardinality: RelationCardinality = Field(..., description="Cardinality between the source and target entities")
    constraints: Optional[List[RelationConstraint]] = Field(None, description="List of constraints that apply to the relation")
    weight: Optional[RelationWeight] = Field(None, description="Optional weight or strength of the relation")
    description: Optional[str] = Field(None, description="Optional description of the relation")

class MappingMetadata(BaseModel):
    num_entities: int = Field(..., description="Total number of entities in the mapping")
    num_relations: int = Field(..., description="Total number of relations mapped")
    method: str = Field(..., description="Method used for mapping relations (e.g., 'manual', 'algorithmic')")
    timestamp: Optional[str] = Field(None, description="Timestamp when the mapping process was completed")
    mapping_algorithm: Optional[str] = Field(None, description="Algorithm used in the mapping process, if applicable")

class RelationMappingOutput(BaseModel):
    algorithm: str = Field(..., description="Algorithm used for relation mapping")
    entities: conlist(Entity) = Field(..., description="List of entities involved in the mapping")
    relations: conlist(Relation) = Field(..., description="List of relations between entities")
    metadata: MappingMetadata = Field(..., description="Metadata regarding the relation mapping process")