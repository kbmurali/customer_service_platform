"""
Load sample data into Neo4j, MySQL, and Chroma databases
Connects to databases via Docker Compose and inserts all sample data
"""
import json
import random
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import uuid

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Database imports
from neo4j import GraphDatabase
import mysql.connector
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path)


# Sample data pools (same as generate_sample_data.py)
FIRST_NAMES = ["John", "Jane", "Michael", "Emily", "David", "Sarah", "Robert", "Lisa", "William", "Jennifer",
               "James", "Mary", "Richard", "Patricia", "Thomas", "Linda", "Charles", "Barbara", "Christopher", "Susan"]

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
              "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"]

CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego",
          "Dallas", "San Jose", "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte"]

STATES = ["NY", "CA", "IL", "TX", "AZ", "PA", "FL", "OH", "NC"]

PLAN_TYPES = ["HMO", "PPO", "EPO", "POS"]
POLICY_TYPES = ["INDIVIDUAL", "FAMILY", "GROUP"]
SPECIALTIES = ["Cardiology", "Orthopedics", "Pediatrics", "Internal Medicine", "Family Medicine",
               "Dermatology", "Psychiatry", "Neurology", "Oncology", "Radiology"]

ICD_CODES = [
    ("I10", "Essential hypertension", "MODERATE"),
    ("E11.9", "Type 2 diabetes mellitus", "MODERATE"),
    ("J45.909", "Unspecified asthma", "MILD"),
    ("M54.5", "Low back pain", "MILD"),
    ("I25.10", "Atherosclerotic heart disease", "SEVERE"),
    ("F41.1", "Generalized anxiety disorder", "MILD"),
    ("K21.9", "Gastro-esophageal reflux disease", "MILD"),
    ("M17.11", "Unilateral primary osteoarthritis, right knee", "MODERATE"),
]

CPT_CODES = [
    ("99213", "Office visit, established patient", "OFFICE_VISIT", 150.00, False),
    ("99214", "Office visit, established patient, complex", "OFFICE_VISIT", 200.00, False),
    ("71020", "Chest X-ray", "DIAGNOSTIC", 100.00, False),
    ("80053", "Comprehensive metabolic panel", "LABORATORY", 50.00, False),
    ("29881", "Knee arthroscopy", "SURGICAL", 3500.00, True),
    ("93000", "Electrocardiogram", "DIAGNOSTIC", 75.00, False),
    ("99285", "Emergency department visit", "EMERGENCY", 500.00, False),
]


class DatabaseLoader:
    """Load sample data into all databases"""
    
    def __init__(self):
        # Database connection parameters (from Docker Compose)
        self.neo4j_uri = os.getenv("NEO4J_KG_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_KG_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_KG_PASSWORD", "healthinsurance")
        
        self.mysql_host = os.getenv("MYSQL_HOST", "localhost")
        self.mysql_port = int(os.getenv("MYSQL_PORT", "3306"))
        self.mysql_user = os.getenv("MYSQL_USER", "healthins")
        self.mysql_password = os.getenv("MYSQL_PASSWORD", "healthins_password")
        self.mysql_database = os.getenv("MYSQL_DATABASE", "health_insurance")
        
        self.chroma_host = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")
        
        self.neo4j_driver = None
        self.mysql_conn = None
        self.chroma_client = None
        self.openai_ef = None
    
    def connect_databases(self):
        """Connect to all databases"""
        print("Connecting to databases...")
        
        try:
            # Connect to Neo4j
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            print("-> Connected to Neo4j")
        except Exception as e:
            print(f"-> Failed to connect to Neo4j: {e}")
            raise
        
        try:
            # Connect to MySQL
            self.mysql_conn = mysql.connector.connect(
                host=self.mysql_host,
                port=self.mysql_port,
                user=self.mysql_user,
                password=self.mysql_password,
                database=self.mysql_database
            )
            print("-> Connected to MySQL")
        except Exception as e:
            print(f"-> Failed to connect to MySQL: {e}")
            raise
        
        try:
            # Connect to Chroma
            self.chroma_client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port
            )
            
            # Create OpenAI Embedding
            self.openai_ef = OpenAIEmbeddingFunction(
                api_key=self.openai_api_key,
                model_name="text-embedding-3-small"
            )
            
            print("-> Connected to Chroma and OpenAI Embedding Created")
        except Exception as e:
            print(f"Failed to initiate Chroma requirements: {e}")
            raise
    
    def close_connections(self):
        """Close all database connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            self.neo4j_driver = None
        if self.mysql_conn:
            self.mysql_conn.close()
            self.mysql_conn = None
        print("Database connections closed")
    
    def load_neo4j_knowledge_graph(self, data: Dict[str, Any]):
        """Load data into Neo4j Knowledge Graph"""
        print("\n=== Loading Neo4j Knowledge Graph ===")
        
        with self.neo4j_driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            print("Cleared existing Neo4j data")
            
            # Load Members
            print(f"Loading {len(data['members'])} members...")
            for member in data['members']:
                session.run("""
                    CREATE (m:Member {
                        memberId: $memberId,
                        firstName: $firstName,
                        lastName: $lastName,
                        dateOfBirth: $dateOfBirth,
                        email: $email,
                        phone: $phone,
                        street: $street,
                        city: $city,
                        state: $state,
                        zipCode: $zipCode,
                        enrollmentDate: $enrollmentDate,
                        status: $status
                    })
                """, **member, **member['address'])
            
            # Load Policies
            print(f"Loading {len(data['policies'])} policies...")
            for policy in data['policies']:
                session.run("""
                    CREATE (p:Policy {
                        policyId: $policyId,
                        policyNumber: $policyNumber,
                        policyType: $policyType,
                        planName: $planName,
                        planType: $planType,
                        effectiveDate: $effectiveDate,
                        expirationDate: $expirationDate,
                        status: $status,
                        premium: $premium,
                        deductible: $deductible,
                        outOfPocketMax: $outOfPocketMax
                    })
                """, **policy)
                
                # Create HAS_POLICY relationship
                session.run("""
                    MATCH (m:Member {memberId: $memberId})
                    MATCH (p:Policy {policyId: $policyId})
                    CREATE (m)-[:HAS_POLICY]-> (p)
                """, memberId=policy['memberId'], policyId=policy['policyId'])
            
            # Load Providers
            print(f"Loading {len(data['providers'])} providers...")
            for provider in data['providers']:
                props = {
                    'providerId': provider['providerId'],
                    'npi': provider['npi'],
                    'providerType': provider['providerType'],
                    'specialty': provider['specialty'],
                    'phone': provider['phone'],
                    **provider['address']
                }
                if provider['providerType'] == 'ORGANIZATION':
                    props['organizationName'] = provider['organizationName']
                else:
                    props['firstName'] = provider['firstName']
                    props['lastName'] = provider['lastName']
                
                session.run("""
                    CREATE (p:Provider {
                        providerId: $providerId,
                        npi: $npi,
                        providerType: $providerType,
                        specialty: $specialty,
                        phone: $phone,
                        street: $street,
                        city: $city,
                        state: $state,
                        zipCode: $zipCode
                    })
                """ + (" SET p.organizationName = $organizationName" if provider['providerType'] == 'ORGANIZATION' else " SET p.firstName = $firstName, p.lastName = $lastName"),
                props)
            
            # Load Claims
            print(f"Loading {len(data['claims'])} claims...")
            for claim in data['claims']:
                session.run("""
                    CREATE (c:Claim {
                        claimId: $claimId,
                        claimNumber: $claimNumber,
                        serviceDate: $serviceDate,
                        submissionDate: $submissionDate,
                        status: $status,
                        totalAmount: $totalAmount,
                        paidAmount: $paidAmount,
                        denialReason: $denialReason,
                        processingDate: $processingDate
                    })
                """, **{k: v for k, v in claim.items() if k not in ['procedures', 'diagnoses']})
                
                # Create relationships
                session.run("""
                    MATCH (m:Member {memberId: $memberId})
                    MATCH (c:Claim {claimId: $claimId})
                    CREATE (m)-[:FILED_CLAIM]-> (c)
                """, memberId=claim['memberId'], claimId=claim['claimId'])
                
                session.run("""
                    MATCH (p:Policy {policyId: $policyId})
                    MATCH (c:Claim {claimId: $claimId})
                    CREATE (c)-[:UNDER_POLICY]-> (p)
                """, policyId=claim['policyId'], claimId=claim['claimId'])
                
                session.run("""
                    MATCH (pr:Provider {providerId: $providerId})
                    MATCH (c:Claim {claimId: $claimId})
                    CREATE (c)-[:SERVICED_BY]-> (pr)
                """, providerId=claim['providerId'], claimId=claim['claimId'])
            
            # Load Prior Authorizations
            print(f"Loading {len(data['prior_authorizations'])} prior authorizations...")
            for pa in data['prior_authorizations']:
                session.run("""
                    CREATE (pa:PriorAuthorization {
                        paId: $paId,
                        paNumber: $paNumber,
                        procedureCode: $procedureCode,
                        procedureDescription: $procedureDescription,
                        requestDate: $requestDate,
                        status: $status,
                        urgency: $urgency,
                        approvalDate: $approvalDate,
                        expirationDate: $expirationDate,
                        denialReason: $denialReason
                    })
                """, **pa)
                
                # Create relationships
                session.run("""
                    MATCH (m:Member {memberId: $memberId})
                    MATCH (pa:PriorAuthorization {paId: $paId})
                    CREATE (m)-[:REQUESTED_PA]-> (pa)
                """, memberId=pa['memberId'], paId=pa['paId'])
                
                session.run("""
                    MATCH (pr:Provider {providerId: $providerId})
                    MATCH (pa:PriorAuthorization {paId: $paId})
                    CREATE (pa)-[:REQUESTED_BY]-> (pr)
                """, providerId=pa['providerId'], paId=pa['paId'])
            
            print("-> Neo4j Knowledge Graph loaded successfully")
    
    def load_mysql_data(self, data: Dict[str, Any]):
        """Load data into MySQL"""
        print("\n=== Loading MySQL Data ===")
        
        cursor = self.mysql_conn.cursor()
        
        try:
            # Note: This assumes tables are already created via mysql_schema.sql
            # In production, you would run the schema file first
            
            # Load users (CSRs)
            print("Loading sample CSR users...")
            users = [
                (str(uuid.uuid4()), 'csr1', 'password_hash_1', 'csr1@healthins.com', 'CSR_TIER1', True),
                (str(uuid.uuid4()), 'csr2', 'password_hash_2', 'csr2@healthins.com', 'CSR_TIER2', True),
                (str(uuid.uuid4()), 'supervisor1', 'password_hash_3', 'supervisor@healthins.com', 'CSR_SUPERVISOR', True),
            ]
            
            for user in users:
                try:
                    cursor.execute("""
                        INSERT INTO users (user_id, username, password_hash, email, role, is_active)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, user)
                except mysql.connector.IntegrityError:
                    pass  # User already exists
            
            self.mysql_conn.commit()
            print(f"-> Loaded {len(users)} CSR users")
            
            print("-> MySQL data loaded successfully")
            
        except Exception as e:
            print(f"-> Error loading MySQL data: {e}")
            self.mysql_conn.rollback()
            raise
        finally:
            cursor.close()
    
    def load_chroma_vectors(self, data: Dict[str, Any]):
        """Load data into Chroma vector database"""
        print("\n=== Loading Chroma Vector Database ===")
        
        try:
            # Create or get collections
            policies_col   = self.chroma_client.get_or_create_collection(name="policies")
            procedures_col = self.chroma_client.get_or_create_collection(name="procedures")
            diagnoses_col  = self.chroma_client.get_or_create_collection(name="diagnoses")
            faqs_col       = self.chroma_client.get_or_create_collection(name="faqs")
            guidelines_col = self.chroma_client.get_or_create_collection(name="clinical_guidelines")

            # Load policy documents
            print(f"Loading {len(data['policies'])} policy documents...")
            policy_docs = []
            policy_ids = []
            policy_metadatas = []
            for policy in data['policies']:
                doc = (f"Policy {policy['policyNumber']}: {policy['planName']} ({policy['planType']}). "
                    f"Premium: ${policy['premium']}/month, Deductible: ${policy['deductible']}, "
                    f"Out-of-pocket max: ${policy['outOfPocketMax']}. "
                    f"Effective: {policy['effectiveDate']} to {policy['expirationDate']}.")
                policy_docs.append(doc)
                policy_ids.append(policy['policyId'])
                policy_metadatas.append({
                    'policyNumber': policy['policyNumber'],
                    'planType': policy['planType'],
                    'status': policy['status']
                })
                
            embeddings = self.openai_ef(policy_docs)
            policies_col.add(
                documents=policy_docs,
                embeddings=embeddings,
                ids=policy_ids,
                metadatas=policy_metadatas
            )
            
            print(f"-> Loaded {len(policy_docs)} policy documents")

            # Load procedure documents
            print(f"Loading {len(data['procedures'])} procedure documents...")
            proc_docs = []
            proc_ids = []
            proc_metadatas = []
            
            for proc in data['procedures']:
                doc = (f"CPT Code {proc['cptCode']}: {proc['description']}. "
                    f"Category: {proc['category']}, Average cost: ${proc['averageCost']}. "
                    f"{'Requires prior authorization.' if proc['requiresPriorAuth'] else 'No prior authorization required.'}")
                proc_docs.append(doc)
                proc_ids.append(proc['cptCode'])
                proc_metadatas.append({
                    'category': proc['category'],
                    'requiresPriorAuth': str(proc['requiresPriorAuth'])
                })
                
            embeddings = self.openai_ef(proc_docs)
            procedures_col.add(
                documents=proc_docs,
                embeddings=embeddings,
                ids=proc_ids,
                metadatas=proc_metadatas
            )
            
            print(f"-> Loaded {len(proc_docs)} procedure documents")

            # Load diagnosis documents
            print(f"Loading {len(data['diagnoses'])} diagnosis documents...")
            diag_docs = []
            diag_ids = []
            diag_metadatas = []
            
            for diag in data['diagnoses']:
                doc = (f"ICD-10 Code {diag['icdCode']}: {diag['description']}. "
                    f"Severity: {diag['severity']}, Category: {diag['category']}.")
                diag_docs.append(doc)
                diag_ids.append(diag['icdCode'])
                diag_metadatas.append({
                    'severity': diag['severity'],
                    'category': diag['category']
                })
                
            embeddings = self.openai_ef(diag_docs)
            diagnoses_col.add(
                documents=diag_docs,
                embeddings=embeddings,
                ids=diag_ids,
                metadatas=diag_metadatas
            )
            
            print(f"-> Loaded {len(diag_docs)} diagnosis documents")

            # Load FAQs
            print("Loading sample FAQs...")
            faqs = [
                ("How do I check my coverage?", "You can check your coverage by logging into the member portal or calling customer service. Your policy details include covered services, deductibles, and out-of-pocket maximums."),
                ("What is a prior authorization?", "Prior authorization is approval from your health insurance plan before you receive certain services or medications. It ensures the service is medically necessary and covered under your plan."),
                ("How long does claim processing take?", "Most claims are processed within 30 days of submission. Complex claims may take longer. You can check claim status online or by calling customer service."),
                ("What is the difference between HMO and PPO?", "HMO (Health Maintenance Organization) requires you to use in-network providers and get referrals for specialists. PPO (Preferred Provider Organization) offers more flexibility to see out-of-network providers at a higher cost."),
                ("How do I find an in-network provider?", "Use our online provider directory or call customer service. In-network providers have contracted rates with your insurance, resulting in lower out-of-pocket costs."),
            ]
            
            faq_docs = [f"Q: {q}\nA: {a}" for q, a in faqs]
            
            embeddings = self.openai_ef(faq_docs)
            faqs_col.add(
                documents=faq_docs,
                embeddings=embeddings,
                ids=[f"faq_{i}" for i in range(len(faqs))],
                metadatas=[{'type': 'faq'} for _ in faqs]
            )
            
            print(f"-> Loaded {len(faq_docs)} FAQ documents")

            # Load clinical guidelines
            print("Loading sample clinical guidelines...")
            
            guidelines = [
                ("Knee Arthroscopy Guidelines", "Knee arthroscopy is indicated for diagnostic evaluation and treatment of intra-articular knee pathology. Prior authorization required. Must have failed conservative treatment for 6 weeks including physical therapy and anti-inflammatory medications."),
                ("Emergency Department Visit Guidelines", "Emergency services are covered for conditions that could result in serious health consequences without immediate care. No prior authorization required. Examples include chest pain, severe bleeding, suspected stroke, or severe breathing difficulty."),
                ("Preventive Care Guidelines", "Annual wellness visits and preventive screenings are covered at 100% with no cost-sharing when using in-network providers. Includes annual physical, immunizations, cancer screenings, and cardiovascular disease screening."),
            ]
            
            guideline_docs = [g[1] for g in guidelines]
            
            embeddings = self.openai_ef(guideline_docs)
            guidelines_col.add(
                documents=guideline_docs,
                embeddings=embeddings,
                ids=[f"guideline_{i}" for i in range(len(guidelines))],
                metadatas=[{'title': g[0], 'type': 'clinical_guideline'} for g in guidelines]
            )
            
            print(f"-> Loaded {len(guideline_docs)} guideline documents")

            # Verify collections loaded
            print("\n-> Chroma collections summary:")
            print(f"   policies:          {policies_col.count()} documents")
            print(f"   procedures:        {procedures_col.count()} documents")
            print(f"   diagnoses:         {diagnoses_col.count()} documents")
            print(f"   faqs:              {faqs_col.count()} documents")
            print(f"   clinical_guidelines:{guidelines_col.count()} documents")

            print("-> Chroma vector database loaded successfully")

        except Exception as e:
            print(f"-> Error loading Chroma data: {e}")
            import traceback
            traceback.print_exc()
            raise


def generate_members(count: int = 50) ->  List[Dict[str, Any]]:
    """Generate member data"""
    members = []
    for i in range(count):
        member_id = str(uuid.uuid4())
        enrollment_date = datetime.now() - timedelta(days=random.randint(30, 1095))
        
        member = {
            "memberId": member_id,
            "firstName": random.choice(FIRST_NAMES),
            "lastName": random.choice(LAST_NAMES),
            "dateOfBirth": (datetime.now() - timedelta(days=random.randint(18*365, 75*365))).strftime("%Y-%m-%d"),
            "email": f"member{i}@example.com",
            "phone": f"555-{random.randint(1000, 9999)}",
            "address": {
                "street": f"{random.randint(100, 9999)} Main St",
                "city": random.choice(CITIES),
                "state": random.choice(STATES),
                "zipCode": f"{random.randint(10000, 99999)}",
                "country": "USA"
            },
            "enrollmentDate": enrollment_date.strftime("%Y-%m-%d"),
            "status": random.choice(["ACTIVE", "ACTIVE", "ACTIVE", "INACTIVE"])
        }
        members.append(member)
    
    return members


def generate_policies(members: List[Dict], count: int = 60) ->  List[Dict[str, Any]]:
    """Generate policy data"""
    policies = []
    for i in range(count):
        member = random.choice(members)
        effective_date = datetime.strptime(member["enrollmentDate"], "%Y-%m-%d")
        expiration_date = effective_date + timedelta(days=365)
        
        policy = {
            "policyId": str(uuid.uuid4()),
            "policyNumber": f"POL-{random.randint(100000, 999999)}",
            "memberId": member["memberId"],
            "policyType": random.choice(POLICY_TYPES),
            "planName": f"{random.choice(PLAN_TYPES)} Plan {random.choice(['Gold', 'Silver', 'Bronze'])}",
            "planType": random.choice(PLAN_TYPES),
            "effectiveDate": effective_date.strftime("%Y-%m-%d"),
            "expirationDate": expiration_date.strftime("%Y-%m-%d"),
            "status": "ACTIVE" if expiration_date > datetime.now() else "EXPIRED",
            "premium": round(random.uniform(200, 800), 2),
            "deductible": random.choice([500, 1000, 2000, 5000]),
            "outOfPocketMax": random.choice([5000, 7500, 10000])
        }
        policies.append(policy)
    
    return policies


def generate_providers(count: int = 30) ->  List[Dict[str, Any]]:
    """Generate provider data"""
    providers = []
    for i in range(count):
        is_org = random.random() < 0.3
        
        provider = {
            "providerId": str(uuid.uuid4()),
            "npi": f"{random.randint(1000000000, 9999999999)}",
            "providerType": "ORGANIZATION" if is_org else "INDIVIDUAL",
            "specialty": random.choice(SPECIALTIES),
            "phone": f"555-{random.randint(1000, 9999)}",
            "address": {
                "street": f"{random.randint(100, 9999)} Medical Plaza",
                "city": random.choice(CITIES),
                "state": random.choice(STATES),
                "zipCode": f"{random.randint(10000, 99999)}",
                "country": "USA"
            }
        }
        
        if is_org:
            provider["organizationName"] = f"{random.choice(LAST_NAMES)} Medical Center"
        else:
            provider["firstName"] = random.choice(FIRST_NAMES)
            provider["lastName"] = random.choice(LAST_NAMES)
        
        providers.append(provider)
    
    return providers


def generate_claims(members: List[Dict], policies: List[Dict], providers: List[Dict], count: int = 100) ->  List[Dict[str, Any]]:
    """Generate claim data"""
    claims = []
    for i in range(count):
        member = random.choice(members)
        member_policies = [p for p in policies if p["memberId"] == member["memberId"]]
        if not member_policies:
            continue
        
        policy = random.choice(member_policies)
        provider = random.choice(providers)
        
        service_date = datetime.now() - timedelta(days=random.randint(1, 180))
        submission_date = service_date + timedelta(days=random.randint(1, 14))
        
        num_procedures = random.randint(1, 3)
        procedures = random.sample(CPT_CODES, min(num_procedures, len(CPT_CODES)))
        diagnoses = random.sample(ICD_CODES, min(random.randint(1, 2), len(ICD_CODES)))
        
        total_amount = sum(p[3] for p in procedures)
        status = random.choice(["SUBMITTED", "UNDER_REVIEW", "APPROVED", "APPROVED", "DENIED"])
        
        claim = {
            "claimId": str(uuid.uuid4()),
            "claimNumber": f"CLM-{random.randint(100000, 999999)}",
            "memberId": member["memberId"],
            "policyId": policy["policyId"],
            "providerId": provider["providerId"],
            "serviceDate": service_date.strftime("%Y-%m-%d"),
            "submissionDate": submission_date.strftime("%Y-%m-%d"),
            "status": status,
            "totalAmount": round(total_amount, 2),
            "paidAmount": round(total_amount * 0.8, 2) if status == "APPROVED" else 0,
            "denialReason": "Not medically necessary" if status == "DENIED" else None,
            "processingDate": (submission_date + timedelta(days=random.randint(3, 21))).strftime("%Y-%m-%d") if status in ["APPROVED", "DENIED"] else None,
            "procedures": [{"cptCode": p[0], "description": p[1]} for p in procedures],
            "diagnoses": [{"icdCode": d[0], "description": d[1]} for d in diagnoses]
        }
        claims.append(claim)
    
    return claims


def generate_prior_authorizations(members: List[Dict], policies: List[Dict], providers: List[Dict], count: int = 50) ->  List[Dict[str, Any]]:
    """Generate prior authorization data"""
    pas = []
    for i in range(count):
        member = random.choice(members)
        member_policies = [p for p in policies if p["memberId"] == member["memberId"]]
        if not member_policies:
            continue
        
        policy = random.choice(member_policies)
        provider = random.choice(providers)
        
        pa_procedures = [p for p in CPT_CODES if p[4]]
        if not pa_procedures:
            continue
        
        procedure = random.choice(pa_procedures)
        
        request_date = datetime.now() - timedelta(days=random.randint(1, 90))
        status = random.choice(["PENDING", "APPROVED", "APPROVED", "DENIED", "EXPIRED"])
        
        pa = {
            "paId": str(uuid.uuid4()),
            "paNumber": f"PA-{random.randint(100000, 999999)}",
            "memberId": member["memberId"],
            "policyId": policy["policyId"],
            "providerId": provider["providerId"],
            "procedureCode": procedure[0],
            "procedureDescription": procedure[1],
            "requestDate": request_date.strftime("%Y-%m-%d"),
            "status": status,
            "urgency": random.choice(["ROUTINE", "ROUTINE", "URGENT", "EMERGENCY"]),
            "approvalDate": (request_date + timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d") if status == "APPROVED" else None,
            "expirationDate": (request_date + timedelta(days=90)).strftime("%Y-%m-%d") if status == "APPROVED" else None,
            "denialReason": "Insufficient documentation" if status == "DENIED" else None
        }
        pas.append(pa)
    
    return pas


def main():
    """Main function to generate and load all sample data"""
    print("=" * 60)
    print("Health Insurance AI Platform - Sample Data Loader")
    print("=" * 60)
    
    # Generate sample data
    print("\n=== Generating Sample Data ===")
    members = generate_members(50)
    print(f"-> Generated {len(members)} members")
    
    policies = generate_policies(members, 60)
    print(f"-> Generated {len(policies)} policies")
    
    providers = generate_providers(30)
    print(f"-> Generated {len(providers)} providers")
    
    claims = generate_claims(members, policies, providers, 100)
    print(f"-> Generated {len(claims)} claims")
    
    pas = generate_prior_authorizations(members, policies, providers, 50)
    print(f"-> Generated {len(pas)} prior authorizations")
    
    # Prepare complete dataset
    data = {
        "members": members,
        "policies": policies,
        "providers": providers,
        "claims": claims,
        "prior_authorizations": pas,
        "diagnoses": [{"icdCode": d[0], "description": d[1], "severity": d[2], "category": "GENERAL"} for d in ICD_CODES],
        "procedures": [{"cptCode": p[0], "description": p[1], "category": p[2], "averageCost": p[3], "requiresPriorAuth": p[4]} for p in CPT_CODES]
    }
    
    # Save to JSON for reference
    with open("sample_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"-> Saved sample data to sample_data.json")
    
    # Load into databases
    loader = DatabaseLoader()
    
    try:
        loader.connect_databases()
        
        loader.load_neo4j_knowledge_graph(data)
        loader.load_mysql_data(data)
        loader.load_chroma_vectors(data)
        
        print("\n" + "=" * 60)
        print("-> ALL DATA LOADED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nTotal records loaded:")
        print(f"  - Members: {len(members)}")
        print(f"  - Policies: {len(policies)}")
        print(f"  - Providers: {len(providers)}")
        print(f"  - Claims: {len(claims)}")
        print(f"  - Prior Authorizations: {len(pas)}")
        print(f"  - Total: {len(members) + len(policies) + len(providers) + len(claims) + len(pas)}")
        
    except Exception as e:
        print(f"\n-> Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        loader.close_connections()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
