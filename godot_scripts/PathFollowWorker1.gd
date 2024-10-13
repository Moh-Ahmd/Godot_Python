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
	# Check if the current offset is less than the maximum offset before moving
	if progress < path_length:
		# Increase the offset based on the speed and the frame time
		progress += Speed * delta

		# Ensure we do not exceed the path length
		if progress >= path_length:
			progress = path_length # Stop at the end of the path

		# Record the current position of the worker
		worker1_actual_positions.append(global_transform.origin)
	

