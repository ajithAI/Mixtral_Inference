run="MaxFreq"
./docker_run_mixtral_inference.sh 64 2048 2048 25 100 ${run}
./docker_run_mixtral_inference.sh 96 2048 128 100 400 ${run}
./docker_run_mixtral_inference.sh 1024 128 128 100 400 ${run}
./docker_run_mixtral_inference.sh 1024 128 2048 10 50 ${run}

./docker_run_mixtral_inference.sh 64 2048 1 100 250 ${run}
./docker_run_mixtral_inference.sh 96 2048 1 100 250 ${run}
./docker_run_mixtral_inference.sh 1024 128 1 100 250 ${run}


./docker_run_mixtral_inference.sh 5120 128 128 15 45 ${run}
./docker_run_mixtral_inference.sh 5120 128 2048 2 8 ${run}
./docker_run_mixtral_inference.sh 480 2048 128 30 90 ${run}
./docker_run_mixtral_inference.sh 320 2048 2048 15 35 ${run}

./docker_run_mixtral_inference.sh 5120 128 1 20 40 ${run}
./docker_run_mixtral_inference.sh 480 2048 1 25 50 ${run}
./docker_run_mixtral_inference.sh 320 2048 1 25 100 ${run}
