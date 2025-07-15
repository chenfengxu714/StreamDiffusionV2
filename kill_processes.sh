PIDS=$(ps aux | grep python | grep -v grep | awk '{print $2}')

for PID in $PIDS; do
	    # echo "Killing Python process with PID: $PID"
	    kill -9 $PID
	done

	echo "All Python processes have been terminated."
