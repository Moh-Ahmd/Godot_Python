extends Node3D

@onready var root := $/root/main/crane/MeshInstance3D/height/cylinder_rotation/jib2/SubViewport/Node3D as Node3D


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta: float) -> void:
	root.global_transform = global_transform
	
