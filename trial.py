from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
username = "neo4j"
password = "########"

try:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    print("Connection successful!")
    
    # Try a simple query
    with driver.session() as session:
        result = session.run("RETURN 'Hello, Neo4j!' AS message")
        print(result.single()["message"])
    
    driver.close()
except Exception as e:
    print(f"Connection failed: {str(e)}")
