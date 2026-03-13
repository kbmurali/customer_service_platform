docker stack rm health_insurance

sleep 10

# Search Services Docker Images

docker rmi csip/search-services-mcp-tools:latest

sleep 1

docker build -f Dockerfile.search_mcp_tools -t csip/search-services-mcp-tools:latest ..

sleep 1

docker rmi csip/search-services-a2a-server:latest

sleep 1

docker build -f Dockerfile.search_a2a_server -t csip/search-services-a2a-server:latest ..

sleep 1

# Provider Services Docker Images

docker rmi csip/provider-services-mcp-tools:latest

sleep 1

docker build -f Dockerfile.provider_mcp_tools -t csip/provider-services-mcp-tools:latest ..

sleep 1

docker rmi csip/provider-services-a2a-server:latest

sleep 1

docker build -f Dockerfile.provider_a2a_server -t csip/provider-services-a2a-server:latest ..

sleep 1

# PA Services Docker Images

docker rmi csip/pa-services-mcp-tools:latest

sleep 1

docker build -f Dockerfile.pa_mcp_tools -t csip/pa-services-mcp-tools:latest ..

sleep 1

docker rmi csip/pa-services-a2a-server:latest

sleep 1

docker build -f Dockerfile.pa_a2a_server -t csip/pa-services-a2a-server:latest ..

sleep 1

# Claims Services Docker Images

docker rmi csip/claims-services-mcp-tools:latest

sleep 1

docker build -f Dockerfile.cs_mcp_tools -t csip/claims-services-mcp-tools:latest ..

sleep 1

docker rmi csip/claims-services-a2a-server:latest

sleep 1

docker build -f Dockerfile.cs_a2a_server -t csip/claims-services-a2a-server:latest ..

sleep 1

# Member Services Docker Images

docker rmi csip/member-services-mcp-tools:latest

sleep 1

docker build -f Dockerfile.ms_mcp_tools -t csip/member-services-mcp-tools:latest ..

sleep 1

docker rmi csip/member-services-a2a-server:latest

sleep 1

docker build -f Dockerfile.ms_a2a_server -t csip/member-services-a2a-server:latest ..

sleep 1

# Infrastructure Services Docker Swarm Deploy

docker stack deploy -c docker-compose-v2.yml health_insurance

sleep 1

# Member Services Docker Swarm Deploy

docker stack deploy -c docker-compose-ms-mcp-tools.yml health_insurance

sleep 1

docker stack deploy -c docker-compose-ms-a2a-server.yml health_insurance

sleep 1

# Claims Services Docker Swarm Deploy

docker stack deploy -c docker-compose-cs-mcp-tools.yml health_insurance

sleep 1

docker stack deploy -c docker-compose-cs-a2a-server.yml health_insurance

sleep 1

# PA Services Docker Swarm Deploy

docker stack deploy -c docker-compose-pa-mcp-tools.yml health_insurance

sleep 1

docker stack deploy -c docker-compose-pa-a2a-server.yml health_insurance

sleep 1

# Provider Services Docker Swarm Deploy

docker stack deploy -c docker-compose-provider-mcp-tools.yml health_insurance

sleep 1

docker stack deploy -c docker-compose-provider-a2a-server.yml health_insurance

sleep 1

# Search Services Docker Swarm Deploy

docker stack deploy -c docker-compose-search-mcp-tools.yml health_insurance

sleep 1

docker stack deploy -c docker-compose-search-a2a-server.yml health_insurance

sleep 1

for i in {1..10}; do
  docker stack services health_insurance
  sleep 3
done

echo ">>>>>>>>>> Done ReDeploying...." 

