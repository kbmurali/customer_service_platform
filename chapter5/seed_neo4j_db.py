#%%
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv, find_dotenv

# Walk up parent directories to find .env — works regardless of working directory
load_dotenv(find_dotenv())

#%%
def seed_database():
    if not os.getenv( "NEO4J_URI", None ):
        raise ValueError( "ERROR: Neo4j BOLT URI not set in the runtime environment!!" )
    
    if not os.getenv( "NEO4J_USER", None ):
        raise ValueError( "ERROR: Neo4j USERNAME not set in the runtime environment!!" )
    
    if not os.getenv( "NEO4J_PASSWORD", None ):
        raise ValueError( "ERROR: Neo4j PASSWORD not set in the runtime environment!!" )
    
    URI = os.getenv( "NEO4J_URI" )
    AUTH = ( os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD") )
    
    driver = GraphDatabase.driver( URI, auth=AUTH )
    
    # Define the queries separately
    clear_query = "MATCH (n) DETACH DELETE n"
    
    seed_query = """
    // Create Policies
    CREATE (p1:Policy {policy_id: "POL-GOLD-001", type: "PPO", limit: 10000})
    CREATE (p2:Policy {policy_id: "POL-SILVER-002", type: "HMO", limit: 5000})
    
     // Create Members
    CREATE (m1:Member {member_id: "MEM-001", name: "Alice Smith"})
    CREATE (m2:Member {member_id: "MEM-002", name: "Bob Smith"}) // Dependent
    CREATE (m3:Member {member_id: "MEM-003", name: "Charlie Jones"})

    // Link Members to Policies
    CREATE (m1)-[:INSURED_BY]->(p1)
    CREATE (m2)-[:INSURED_BY]->(p1)
    CREATE (m3)-[:INSURED_BY]->(p2)

    // Create Diagnoses
    CREATE (d1:Diagnosis {code: "E11", description: "Type 2 Diabetes"})
    CREATE (d2:Diagnosis {code: "I10", description: "Hypertension"})

    // Create Claims
    CREATE (c1:Claim {claim_id: "CLM-100", amount: 250.00, status: "Paid", date: "2024-01-15", service: "Endocrinology"})
    CREATE (c2:Claim {claim_id: "CLM-101", amount: 1500.00, status: "Pending", date: "2024-02-20", service: "Cardiology"})
    CREATE (c3:Claim {claim_id: "CLM-102", amount: 100.00, status: "Denied", date: "2024-03-05", service: "General Practice"})
    CREATE (c4:Claim {claim_id: "CLM-103", amount: 250.00, status: "Paid", date: "2024-05-03", service: "Endocrinology"})

    // Link Claims
    CREATE (m1)-[:FILED]->(c1)
    CREATE (c1)-[:FOR_DIAGNOSIS]->(d1)

    CREATE (m2)-[:FILED]->(c2)
    CREATE (c2)-[:FOR_DIAGNOSIS]->(d2)

    CREATE (m3)-[:FILED]->(c3)
    CREATE (c3)-[:FOR_DIAGNOSIS]->(d2)
    
    CREATE (m3)-[:FILED]->(c4)
    CREATE (c4)-[:FOR_DIAGNOSIS]->(d1)
    """
    
    try:
        with driver.session() as session:
            # 1. Execute the Clear Command
            print("Clearing database...")
            session.run(clear_query)
            
            # 2. Execute the Seed Command
            print("Seeding data...")
            session.run(seed_query)
            
            print("Database seeded successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.close()

#%%
seed_database()
# %%
