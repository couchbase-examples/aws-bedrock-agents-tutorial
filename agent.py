# Required imports
import json
import os
from datetime import datetime, timezone
from getpass import getpass
from typing import Any, Dict, List

import boto3
import numpy as np
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, QueryOptions


class CouchbaseHandler:
    def __init__(self,
                 host: str,
                 username: str,
                 password: str,
                 bucket: str,
                 scope: str = '_default',
                 collection: str = None):
        
        # Validate required parameters
        required_params = {
            'host': host,
            'username': username,
            'password': password,
            'bucket': bucket,
            'collection': collection
        }
        
        missing_params = [k for k, v in required_params.items() if not v]
        if missing_params:
            raise ValueError(f"Missing required Couchbase parameters: {', '.join(missing_params)}")
        
        # Initialize Couchbase connection
        auth = PasswordAuthenticator(username, password)
        cluster_opts = ClusterOptions(auth)
        self.cluster = Cluster(f'couchbase://{host}', cluster_opts)
        self.bucket = self.cluster.bucket(bucket)
        self.scope = self.bucket.scope(scope)
        self.collection = self.scope.collection(collection)

    def store_document(self, document_id: str, content: str, embedding: List[float]) -> None:
        """Store document and its embedding in Couchbase"""
        document = {
            'content': content,
            'embedding': embedding,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        self.collection.upsert(document_id, document)

    def vector_search(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        """Perform vector similarity search in Couchbase"""
        query = f"""
        SELECT content, 
               vector_distance(embedding, $query_embedding) as similarity
        FROM `{self.bucket.name}`.`{self.scope.name}`.`{self.collection.name}`
        ORDER BY similarity
        LIMIT $limit
        """
        
        params = {
            'query_embedding': query_embedding,
            'limit': limit
        }
        
        results = self.cluster.query(
            query,
            QueryOptions(named_parameters=params)
        )
        
        return [row for row in results]

    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a N1QL query with parameters"""
        results = self.cluster.query(
            query,
            QueryOptions(named_parameters=parameters or {})
        )
        return [row for row in results]

class BedrockHandler:
    def __init__(self, region: str, aws_access_key_id: str, aws_secret_access_key: str):
        # Initialize AWS clients with provided credentials
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        self.bedrock_agent = boto3.client(
            service_name='bedrock-agent-runtime',
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        
        self.lambda_client = boto3.client(
            service_name='lambda',
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

    def create_embedding(self, text: str) -> List[float]:
        """Create vector embeddings using Bedrock's embedding model"""
        embedding_prompt = {
            "inputText": text,
            "modelId": "amazon.titan-embed-text-v1"
        }
        
        response = self.bedrock.invoke_model(
            modelId='amazon.titan-embed-text-v1',
            body=json.dumps(embedding_prompt)
        )
        
        embedding_data = json.loads(response['body'].read())
        return embedding_data['embedding']

    def invoke_claude(self, prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> Dict:
        """Invoke Claude model with given parameters"""
        request = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = self.bedrock.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20241022',
            body=json.dumps(request)
        )
        
        return json.loads(response['body'].read())

    def invoke_lambda(self, function_name: str, payload: Dict) -> Dict:
        """Invoke AWS Lambda function"""
        response = self.lambda_client.invoke(
            FunctionName=function_name,
            Payload=json.dumps(payload)
        )
        return json.loads(response['Payload'].read())

class AgentHandler:
    def __init__(self, couchbase_handler: CouchbaseHandler, bedrock_handler: BedrockHandler):
        self.couchbase = couchbase_handler
        self.bedrock = bedrock_handler

    def preprocess_input(self, user_input: str) -> Dict:
        """Pre-process prompt to structure and validate user input"""
        response = self.bedrock.invoke_claude(
            prompt=f"Process this user request and convert to structured format: {user_input}"
        )
        return response

    def orchestrate_action(self, structured_input: Dict) -> Dict:
        """Determine actions based on structured input"""
        response = self.bedrock.invoke_claude(
            prompt=f"Determine required actions for this request: {json.dumps(structured_input)}"
        )
        return response

    def generate_response(self, action_results: Dict, context: Dict) -> str:
        """Generate final response using knowledge base and action results"""
        response = self.bedrock.invoke_claude(
            prompt=f"""
            Generate user response based on:
            Action Results: {json.dumps(action_results)}
            Context: {json.dumps(context)}
            """,
            max_tokens=1000,
            temperature=0.7
        )
        return response['completion']

    def execute_actions(self, action_plan: Dict) -> Dict:
        """Execute the planned actions"""
        results = {}
        
        for action in action_plan.get('actions', []):
            if action['type'] == 'database_query':
                results[action['id']] = self.couchbase.execute_query(
                    action['query'],
                    action.get('parameters', {})
                )
            elif action['type'] == 'api_call':
                results[action['id']] = self.bedrock.invoke_lambda(
                    action['function_name'],
                    action['payload']
                )
        
        return results

    def process_user_request(self, user_input: str) -> str:
        """Main method to process user requests through the entire pipeline"""
        try:
            # Step 1: Pre-process input
            structured_input = self.preprocess_input(user_input)
            
            # Step 2: Create embedding for vector search
            query_embedding = self.bedrock.create_embedding(user_input)
            
            # Step 3: Search vector store for relevant context
            relevant_docs = self.couchbase.vector_search(query_embedding)
            
            # Step 4: Determine required actions
            action_plan = self.orchestrate_action(structured_input)
            
            # Step 5: Execute actions and gather results
            action_results = self.execute_actions(action_plan)
            
            # Step 6: Generate final response
            context = {
                'relevant_docs': relevant_docs,
                'structured_input': structured_input
            }
            
            final_response = self.generate_response(action_results, context)
            
            return final_response
            
        except Exception as e:
            return f"Error processing request: {str(e)}"

def get_credentials():
    """Get credentials from environment variables or user input"""
    # Couchbase credentials
    couchbase_host = os.getenv('COUCHBASE_HOST') or input("Enter Couchbase host: ")
    couchbase_username = os.getenv('COUCHBASE_USERNAME') or input("Enter Couchbase username: ")
    couchbase_password = os.getenv('COUCHBASE_PASSWORD') or getpass("Enter Couchbase password: ")
    couchbase_bucket = os.getenv('COUCHBASE_BUCKET') or input("Enter Couchbase bucket: ")
    couchbase_scope = os.getenv('COUCHBASE_SCOPE') or input("Enter Couchbase scope (default: _default): ") or '_default'
    couchbase_collection = os.getenv('COUCHBASE_COLLECTION') or input("Enter Couchbase collection: ")

    # AWS credentials
    aws_region = os.getenv('AWS_REGION') or input("Enter AWS region (default: us-east-1): ") or 'us-east-1'
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID') or input("Enter AWS access key ID: ")
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY') or getpass("Enter AWS secret access key: ")

    return {
        'couchbase': {
            'host': couchbase_host,
            'username': couchbase_username,
            'password': couchbase_password,
            'bucket': couchbase_bucket,
            'scope': couchbase_scope,
            'collection': couchbase_collection
        },
        'aws': {
            'region': aws_region,
            'access_key_id': aws_access_key_id,
            'secret_access_key': aws_secret_access_key
        }
    }

# Example usage
def main():
    # Get credentials
    creds = get_credentials()
    
    # Initialize handlers
    couchbase_handler = CouchbaseHandler(
        host=creds['couchbase']['host'],
        username=creds['couchbase']['username'],
        password=creds['couchbase']['password'],
        bucket=creds['couchbase']['bucket'],
        scope=creds['couchbase']['scope'],
        collection=creds['couchbase']['collection']
    )
    
    bedrock_handler = BedrockHandler(
        region=creds['aws']['region'],
        aws_access_key_id=creds['aws']['access_key_id'],
        aws_secret_access_key=creds['aws']['secret_access_key']
    )
    
    # Initialize the agent handler
    agent = AgentHandler(couchbase_handler, bedrock_handler)
    
    # Example: Process a user request
    user_input = "What were our sales numbers for Q1 2024?"
    response = agent.process_user_request(user_input)
    print(f"Response: {response}")

    # Example: Store new document with embedding
    document_content = "Q1 2024 sales report shows 15% growth in North America"
    embedding = agent.bedrock.create_embedding(document_content)
    agent.couchbase.store_document(
        document_id="sales_report_q1_2024",
        content=document_content,
        embedding=embedding
    )

if __name__ == "__main__":
    main()