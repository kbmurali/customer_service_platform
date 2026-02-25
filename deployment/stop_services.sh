docker stack rm health_insurance

for i in {1..5}; do
  docker stack services health_insurance
  sleep 2
done
