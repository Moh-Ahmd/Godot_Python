extends Node3D

# state variables
var current_position = Vector3(-18.286, 0, -5.988)  # Initial position
var velocity = Vector3()  # Initial velocity
var orientation = 0.0  # Initial orientation (radians)
var angular_velocity = 0.0  # (radians per second)

# parameters
var max_speed = 200.0  
var acceleration_rate = 10.0 
var drag_coefficient = 0.02 

# Time step
var dt = 1.0 / 30.0  # (30 FPS)

func _process(delta):
	dt = delta
	handle_input()

	# direction based on orientation
	var forward = Vector3(sin(orientation), 0, cos(orientation))
	
	# Apply acceleration in the forward direction if moving
	if acceleration_rate != 0:
		velocity = forward * acceleration_rate * dt
	else:
		velocity = Vector3()  # Stop

	# Limit velocity to max speed
	if velocity.length() > max_speed:
		velocity = velocity.normalized() * max_speed

	# position update based on velocity
	current_position += velocity * dt

	# orientation update
	orientation += angular_velocity * dt

	# global update position and orientation
	transform.origin = current_position
	rotation.y = orientation

func handle_input():
	# reset acceleration and angular velocity
	acceleration_rate = 0.0
	angular_velocity = 0.0

	# Forward and backward movement
	if Input.is_action_pressed("move_forward"):
		acceleration_rate = max_speed  # max speed for quick acceleration
	elif Input.is_action_pressed("move_backward"):
		acceleration_rate = -max_speed

	# Turning left and right
	if Input.is_action_pressed("turn_left"):
		angular_velocity = -1.0 
	elif Input.is_action_pressed("turn_right"):
		angular_velocity = 1.0



