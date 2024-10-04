extends PathFollow3D


var Speed = 5.0 # Movement speed along the path
var worker1_actual_positions = []  # Array to store positions
var path_length: float





func _ready():
	progress = 0  # Start at the beginning of the path
	loop = false  # Disable looping
	# Ensure we are correctly accessing the Path3D node
	var path_node = get_parent() as Path3D
	if path_node and path_node.curve:
		path_length = path_node.curve.get_baked_length()
	else:
		push_error("Path3D or Curve3D not found or incorrectly set up.")
	

func _process(delta):
	#Speed = randf_range(0.0, 5.0)
	# Check if the current offset is less than the maximum offset before moving
	if progress < path_length:
		# Increase the offset based on the speed and the frame time
		progress += Speed * delta

		# Ensure we do not exceed the path length
		if progress >= path_length:
			progress = path_length # Stop at the end of the path

		# Record the current position of the worker
		worker1_actual_positions.append(global_transform.origin)
	

#func get_worker1_actual_positions():
	#print(worker1_actual_positions)
	#return worker1_actual_positions

#var initial_speed = 0.0  # Starting speed
#var acceleration = 0.5  # Constant acceleration
#var max_speed = 5.0  # Maximum speed
#var speed = initial_speed  # Current speed
#var worker1_actual_positions = []  # Array to store positions
#var path_length: float
#
#func _ready():
	#progress = 0  # Start at the beginning of the path
	#loop = false  # Disable looping
	## Ensure we are correctly accessing the Path3D node
	#var path_node = get_parent() as Path3D
	#if path_node and path_node.curve:
		#path_length = path_node.curve.get_baked_length()
	#else:
		#push_error("Path3D or Curve3D not found or incorrectly set up.")
#
#func _process(delta):
	## Update speed with constant acceleration
	#speed += acceleration * delta
	## Clamp speed to max_speed
	#speed = min(speed, max_speed)
	#
	## Check if the current offset is less than the maximum offset before moving
	#if progress < path_length:
		## Increase the offset based on the speed and the frame time
		#progress += speed * delta
#
		## Ensure we do not exceed the path length
		#if progress >= path_length:
			#progress = path_length  # Stop at the end of the path
#
		## Record the current position of the worker
		#worker1_actual_positions.append(global_transform.origin)
		
		
		
#var initial_speed = 0.0
#var max_speed = 5.0
#var speed = initial_speed
#var worker1_actual_positions = []  # Array to store positions
#var path_length: float
#var k = -2  # Spring constant (negative for restoring force)
#
#func _ready():
	#progress = 0
	#loop = false
	#var path_node = get_parent() as Path3D
	#if path_node and path_node.curve:
		#path_length = path_node.curve.get_baked_length()
	#else:
		#push_error("Path3D or Curve3D not found or incorrectly set up.")
#
#func _process(delta):
	#var acceleration = k * global_transform.origin.x  # Spring-like acceleration
	#speed += acceleration * delta
	#speed = clamp(speed, -max_speed, max_speed)  # Clamping the speed to max limits
	#
	#if progress < path_length:
		#progress += speed * delta
		#if progress >= path_length:
			#progress = path_length
		#
		#worker1_actual_positions.append(global_transform.origin)
		
		
#var initial_speed = 0.0
#var max_speed = 10.0
#var min_speed = -10.0
#var speed = initial_speed
#var worker1_actual_positions = []  # Array to store positions
#var path_length: float
#var random_direction_factor = 0.2  # Factor to introduce randomness in direction
#
#func _ready():
	#progress = 0
	#loop = false
	#var path_node = get_parent() as Path3D
	#if path_node and path_node.curve:
		#path_length = path_node.curve.get_baked_length()
	#else:
		#push_error("Path3D or Curve3D not found or incorrectly set up.")
#
#func _process(delta):
	#var random_acceleration = (randf() * 11.0 - 1.0) * random_direction_factor  # Random acceleration component
	#speed += random_acceleration
	#speed = clamp(speed, min_speed, max_speed)  # Clamping the speed to max limits
	#
	## Direction change logic
	#if randf() < 0.1:  # 10% chance every frame to potentially reverse direction
		#speed = -speed
	#
	#if progress < path_length:
		#progress += speed * delta
		#if progress >= path_length:
			#progress = path_length
		#
		#worker1_actual_positions.append(global_transform.origin)
#
	## Log the position and speed for analysis
	#print("Position: ", global_transform.origin, " Speed: ", speed)
