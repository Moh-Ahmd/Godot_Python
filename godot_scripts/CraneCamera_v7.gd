extends Camera3D

signal detections_updated(detections)

var frame_socket = StreamPeerTCP.new()
var result_socket = StreamPeerTCP.new()
var server_host = "127.0.0.1"
var frame_port = 9999
var result_port = 9998
var is_frame_server_connected = false
var is_result_server_connected = false
var frame_interval = 1.0 / 30.0
var last_frame_time = 0
var frame_number = 1
var frames_sent = 0
var frames_received = 0
var detection_buffer = ""

func _ready():
	set_process(true)
	_connect_to_servers()

func _process(delta):
	if is_frame_server_connected:
		frame_socket.poll()
		if Time.get_ticks_msec() - last_frame_time > frame_interval * 1000:
			_capture_and_send_frame()
			last_frame_time = Time.get_ticks_msec()
	
	if is_result_server_connected:
		result_socket.poll()
		#_check_for_json_ready()
		_receive_detection_result()
func _connect_to_servers():
	print("Attempting to connect to frame server at: ", server_host, ":", frame_port)
	var err = frame_socket.connect_to_host(server_host, frame_port)
	if err == OK:
		print("Connected to frame server")
		is_frame_server_connected = true
	else:
		print("Failed to connect to frame server: ", err)
	
	print("Attempting to connect to result server at: ", server_host, ":", result_port)
	err = result_socket.connect_to_host(server_host, result_port)
	if err == OK:
		print("Connected to result server")
		is_result_server_connected = true
	else:
		print("Failed to connect to result server: ", err)

func _capture_and_send_frame():
	var subviewport = get_parent().get_parent().get_parent().get_parent()
	var img = subviewport.get_texture().get_image()
	var buffer = img.get_data()
	print("FPS: ", Engine.get_frames_per_second())
	# Send frame number
	var frame_number_bytes = PackedByteArray()
	frame_number_bytes.resize(4)
	frame_number_bytes[0] = (frame_number >> 24) & 0xFF
	frame_number_bytes[1] = (frame_number >> 16) & 0xFF
	frame_number_bytes[2] = (frame_number >> 8) & 0xFF
	frame_number_bytes[3] = frame_number & 0xFF
	frame_socket.put_data(frame_number_bytes)
	
	# Send frame buffer
	frame_socket.put_data(buffer)
	
	print("Sent frame number: ", frame_number)
	frame_number += 1
	frames_sent += 1

func _process_detection_buffer():
	while true:
		var start_index = detection_buffer.find("<START>")
		var end_index = detection_buffer.find("<END>")
		
		if start_index != -1 and end_index != -1 and start_index < end_index:
			var detection_json = detection_buffer.substr(start_index + 7, end_index - start_index - 7)
			#print("Extracted JSON: ", detection_json)
			var detection_result = JSON.parse_string(detection_json)
			
			if detection_result and "detections" in detection_result:
				#print("Received detections for frame: ", detection_result.get("frame_number", "unknown"))
				emit_signal("detections_updated", detection_result['detections'])
				frames_received += 1
			else:
				print("Received invalid detection result")
			
			detection_buffer = detection_buffer.substr(end_index + 5)
		else:
			break  # No complete message in buffer, wait for more data

func _receive_detection_result():
	var json_byte_len = result_socket.get_data(4)	
	var json_pb_array = json_byte_len[1]
	#unpack big-endian
	var json_len = 0
	json_len += json_pb_array[0] << 24  
	json_len += json_pb_array[1] << 16  
	json_len += json_pb_array[2] << 8  
	json_len += json_pb_array[3]  
	print("Expected message length: ", json_len)
	#print(json_len)
	#initialize an empty PackedByteArray 
	var json_data = PackedByteArray()
	while len(json_data) < json_len:
		var json_chunk = result_socket.get_partial_data(json_len - len(json_data))[1]
		if not json_chunk:
			break
		json_data += json_chunk
		var json_string = json_data.get_string_from_utf8()
		print("Received message: ", json_string)
		detection_buffer += json_string
		#print("Received detection result for frame: ", detection_buffer.get("frame_number", "unknown"))
		frames_received += 1
		_process_detection_buffer()


#func _exit_tree():
	#if is_frame_server_connected:
		#frame_socket.disconnect_from_host()
		##print("Disconnected from frame server")
	#if is_result_server_connected:
		#result_socket.disconnect_from_host()
		##print("Disconnected from result server")
	##print("Final statistics:")
	##print("Total frames sent: ", frames_sent)
	##print("Total frames with detections received: ", frames_received)
func _exit_tree():
	if is_frame_server_connected:
		frame_socket.put_data("<SHUTDOWN>".to_utf8_buffer())
		frame_socket.disconnect_from_host()
		print("Disconnected from frame server")
	if is_result_server_connected:
		result_socket.put_data("<SHUTDOWN>".to_utf8_buffer())
		result_socket.disconnect_from_host()
		print("Disconnected from result server")
	print("Final statistics:")
	print("Total frames sent: ", frames_sent)
	print("Total frames with detections received: ", frames_received)
