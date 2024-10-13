extends PathFollow3D

const speed = 5.0  

var moving_forward: bool = false
var moving_backward: bool = false
var path_length: float

func _ready():
	#start at the beginning of path
	progress = 0
	#no looping!  
	loop = false
	#get the path to follow
	var path_node = get_parent() as Path3D
	if path_node and path_node.curve:
		path_length = path_node.curve.get_baked_length()

func _process(delta):
	#get user input
	check_input()

	#camera-movement based on input
	if moving_forward:
		#if progress (position relative to path length) is
		#less than path_length and user input "ui_up"
		if progress < path_length:
			#increase progress by speed * delta
			progress += speed * delta
		#if reached the end of path
		elif progress >= path_length:
			#stop
			progress = 1.0 
			
	#same goes for user's input "ui_down"		
	elif moving_backward:
		if progress < path_length:
			progress -= speed * delta
		elif progress <= path_length:

			progress = 0.0  # Stop at the start of the path
#function to take user's input
func check_input():
	moving_forward = Input.is_action_pressed("ui_up")
	moving_backward = Input.is_action_pressed("ui_down")
