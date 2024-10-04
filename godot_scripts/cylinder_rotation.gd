extends Node3D

var rotation_speed = 0.02
var rotation_dir = 0.0



func _process(delta):
	#rotation input:
	#ui_left: keyboard arrow to the left, s. Key mappings(should be added in a seperate readme.txt later!)
	if Input.is_action_pressed("ui_left"): 
		rotation_dir -= 1
	#ui_right: to the right
	elif Input.is_action_pressed("ui_right"): 
		rotation_dir += 1
	#if no input, stop
	else:
		rotation_dir = 0
	
	#apply rotation around the up-axis y
	rotate_y(rotation_dir * rotation_speed * delta)
		
	
	
	
