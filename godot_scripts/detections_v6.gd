extends Node2D

var tcp_socket = StreamPeerTCP.new()
var server_host = "127.0.0.1"
var server_port = 10000
var is_connected = false
var frame_interval = 1.0 / 30.0
var last_frame_time = 0
var det_node
var detections = []
var previous_positions = {}
var previous_velocities = {}
var previous_times = {}
var line_nodes = []
var label_nodes = []
var frame_counter = 0
var measurement_buffer = []
var previous_excavator_position = Vector3.ZERO
var previous_time = 0.0

func _ready():
	det_node = get_node("/root/main/crane/MeshInstance3D/height/cylinder_rotation/jib2/SubViewport/Node3D/CraneCameraPath/PathFollow3D/CraneCamera")
	det_node.detections_updated.connect(_on_detections_updated)
	connect_t_server()

func _process(delta):
	if is_connected:
		tcp_socket.poll()
		if Time.get_ticks_msec() - last_frame_time > frame_interval * 1000:
			_process_measurements()
			_send_measurements()
			last_frame_time = Time.get_ticks_msec()
			frame_counter += 1
			if frame_counter % 30 == 0:  # Print every second
				print("Frames processed: ", frame_counter)

func connect_t_server():
	var err = tcp_socket.connect_to_host(server_host, server_port)
	if err == OK:
		print("Client 2 Connected to server")
		is_connected = true
	else:
		print("Failed to connect: ", err)

func _on_detections_updated(new_detections):
	detections = new_detections
	print("Received new detections: ", new_detections.size())
	while new_detections.size() > line_nodes.size():
		var line = Line2D.new()
		line.width = 2
		line.default_color = Color(1, 0, 0)  # Red color for bounding box
		add_child(line)
		line_nodes.append(line)

		var label_node = Label.new()
		add_child(label_node)
		label_nodes.append(label_node)

	for i in range(new_detections.size(), line_nodes.size()):
		line_nodes[i].hide()
		label_nodes[i].hide()

	for i in range(new_detections.size()):
		var detection = new_detections[i]
		var bbox = detection["bbox"]
		var label_text = detection["label"]
		var confidence = detection["confidence"]

		var line = line_nodes[i]
		line.points = [
			Vector2(bbox[0], bbox[1]),
			Vector2(bbox[2], bbox[1]),
			Vector2(bbox[2], bbox[3]),
			Vector2(bbox[0], bbox[3]),
			Vector2(bbox[0], bbox[1])
		]
		line.show()

		var label_node = label_nodes[i]
		label_node.text = "%s: %.2f" % [label_text, confidence]
		label_node.set_position(Vector2(bbox[0], bbox[1] - 20))
		label_node.add_theme_font_override("font", ThemeDB.fallback_font)
		label_node.show()

	queue_redraw()

func pixel_to_world(x_pixel, y_pixel):
	var camera = get_node("/root/main/crane/MeshInstance3D/height/cylinder_rotation/jib2/SubViewport/Node3D/CraneCameraPath/PathFollow3D/CraneCamera")
	var viewport_size = camera.get_viewport().size
	var normalized_x = (2.0 * x_pixel / viewport_size.x) - 1.0
	var normalized_y = 1.0 - (2.0 * y_pixel / viewport_size.y)
	var ray_origin = camera.project_ray_origin(Vector2(x_pixel, y_pixel))
	var ray_direction = camera.project_ray_normal(Vector2(x_pixel, y_pixel))
	
	if ray_direction.y == 0:
		return null  # Ray is parallel to the ground plane, no intersection

	var t = -ray_origin.y / ray_direction.y
	var intersection_point = ray_origin + ray_direction * t
	var vector = Vector3(intersection_point.x, 0.0, intersection_point.z)

	return vector

func _process_measurements():
	var coords_dict = {}
	var current_time = Time.get_ticks_msec() / 1000.0

	# Get true positions and velocities
	var excavator_node = get_node("/root/main/excavator")
	var global_position_excavator = excavator_node.global_transform.origin
	var global_x_excavator = global_position_excavator.x
	var global_z_excavator = global_position_excavator.z

	# Calculate excavator velocity based on position change
	var velocity_x_excavator = 0
	var velocity_z_excavator = 0
	if previous_excavator_position != Vector3.ZERO:
		var delta_time = current_time - previous_time
		if delta_time > 0:
			velocity_x_excavator = (global_x_excavator - previous_excavator_position.x) / delta_time
			velocity_z_excavator = (global_z_excavator - previous_excavator_position.z) / delta_time

	# Update previous position and time for next frame
	previous_excavator_position = global_position_excavator
	previous_time = current_time
	
	var worker_node = get_node("/root/main/Path_Worker1/PathFollow3D/worker_1")
	var global_position_worker = worker_node.global_position
	var global_x_worker = global_position_worker.x
	var global_z_worker = global_position_worker.z
	var velocity_worker = worker_node.velocity
	var velocity_x_worker = velocity_worker.x
	var velocity_z_worker = velocity_worker.z

	for detection in detections:
		var label = detection["label"]
		if label != "worker" and label != "excavator":
			continue

		var center = detection["center"]
		var world_coords = pixel_to_world(center[0], center[1])
		if world_coords == null:
			continue

		var float_array = PackedFloat32Array()
		
		# Detected position
		float_array.append(world_coords.x)
		float_array.append(world_coords.z)

		# True position and velocity
		if label == "worker":
			float_array.append(global_x_worker)
			float_array.append(global_z_worker)
			float_array.append(velocity_x_worker)
			float_array.append(velocity_z_worker)
		elif label == "excavator":
			float_array.append(global_x_excavator)
			float_array.append(global_z_excavator)
			float_array.append(velocity_x_excavator)
			float_array.append(velocity_z_excavator)

		# Calculate measured velocity and acceleration
		if not label in previous_positions:
			previous_positions[label] = Vector2(world_coords.x, world_coords.z)
			previous_velocities[label] = Vector2(0, 0)
			previous_times[label] = current_time
			# Append zero velocity and acceleration for the first measurement
			float_array.append(0)
			float_array.append(0)
			float_array.append(0)
			float_array.append(0)
		else:
			var delta_time = current_time - previous_times[label]
			var velocity = Vector2()
			var acceleration = Vector2()

			if delta_time > 0:
				velocity.x = (world_coords.x - previous_positions[label].x) / delta_time
				velocity.y = (world_coords.z - previous_positions[label].y) / delta_time
				acceleration.x = (velocity.x - previous_velocities[label].x) / delta_time
				acceleration.y = (velocity.y - previous_velocities[label].y) / delta_time

			# Append measured velocity and acceleration
			float_array.append(velocity.x)
			float_array.append(velocity.y)
			float_array.append(acceleration.x)
			float_array.append(acceleration.y)

			previous_positions[label] = Vector2(world_coords.x, world_coords.z)
			previous_velocities[label] = velocity
			previous_times[label] = current_time

		# Ensure the array has 10 elements
		while float_array.size() < 10:
			float_array.append(0)

		coords_dict[label] = [float_array]

	if coords_dict:
		var buffer_string = JSON.stringify(coords_dict)
		var buffer = buffer_string.to_utf8_buffer()
		measurement_buffer.append(buffer)
		
func _send_measurements():
	if measurement_buffer.size() > 0:
		var buffer = measurement_buffer.pop_front()
		var buffer_size = buffer.size()

		var packed_buffer_size = PackedByteArray()
		packed_buffer_size.resize(4)
		packed_buffer_size[0] = (buffer_size >> 24) & 0xFF
		packed_buffer_size[1] = (buffer_size >> 16) & 0xFF
		packed_buffer_size[2] = (buffer_size >> 8) & 0xFF
		packed_buffer_size[3] = buffer_size & 0xFF
		
		tcp_socket.put_data(packed_buffer_size)
		var msg_bytes = tcp_socket.get_data(10)
		var stat_code = msg_bytes[0]
		var actual_msg_size = msg_bytes[1]
		var size_msg = ""
		for m in actual_msg_size:
			size_msg += String.chr(m)
		if size_msg == "<ACK_SIZE>":
			tcp_socket.put_data(buffer)

func _exit_tree():
	if is_connected:
		tcp_socket.put_data("<SHUTDOWN>".to_utf8_buffer())
		tcp_socket.disconnect_from_host()
		print("Disconnected from measurement server")
	print("Frames processed: ", frame_counter)

