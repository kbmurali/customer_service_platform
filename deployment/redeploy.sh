docker stack rm health_insurance

sleep 10

docker rmi csip/member-services-mcp-tools:latest

sleep 5

docker build -f Dockerfile.ms_mcp_tools -t csip/member-services-mcp-tools:latest ..

sleep 2

docker rmi csip/member-services-a2a-server:latest

sleep 5

docker build -f Dockerfile.ms_a2a_server -t csip/member-services-a2a-server:latest ..

sleep 5

docker stack deploy -c docker-compose-v2.yml health_insurance

sleep 10

docker stack deploy -c docker-compose-ms-mcp-tools.yml health_insurance

sleep 5

docker stack deploy -c docker-compose-ms-a2a-server.yml health_insurance

for i in {1..5}; do
  docker stack services health_insurance
  sleep 5
done

echo ">>>>>>>>>> Done ReDeploying...." 

