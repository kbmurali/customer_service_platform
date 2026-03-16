docker stack deploy -c docker-compose-v2.yml health_insurance

sleep 5

docker stack deploy -c docker-compose-ms-mcp-tools.yml health_insurance

sleep 1

docker stack deploy -c docker-compose-ms-a2a-server.yml health_insurance

sleep 1

docker stack deploy -c docker-compose-cs-mcp-tools.yml health_insurance

sleep 1

docker stack deploy -c docker-compose-cs-a2a-server.yml health_insurance

sleep 1

docker stack deploy -c docker-compose-pa-mcp-tools.yml health_insurance

sleep 1

docker stack deploy -c docker-compose-pa-a2a-server.yml health_insurance

sleep 1

docker stack deploy -c docker-compose-provider-mcp-tools.yml health_insurance

sleep 1

docker stack deploy -c docker-compose-provider-a2a-server.yml health_insurance

sleep 1

docker stack deploy -c docker-compose-search-mcp-tools.yml health_insurance

sleep 1

docker stack deploy -c docker-compose-search-a2a-server.yml health_insurance

sleep 10

docker stack deploy -c docker-compose-agentic-access.yml health_insurance

sleep 1

for i in {1..10}; do
  docker stack services health_insurance
  sleep 3
done

echo ">>>>>>>>>> Done Deploying...." 