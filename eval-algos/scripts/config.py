GCP_CREDENTIALS = 'credentials.json'         # UPDATE THIS WITH YOUR CREDENTIALS
GCP_PROJECT_ID = 'opensource-observer'       # UPDATE THIS WITH YOUR PROJECT ID
MODELS = [
    'projects_v1',
    'int_superchain_s7_onchain_metrics_by_project',
    'int_superchain_s7_devtooling_metrics_by_project', 
    'int_superchain_s7_onchain_builder_eligibility',
    'int_superchain_s7_trusted_developers',
    'int_superchain_s7_project_to_developer_graph',
    'int_superchain_s7_project_to_dependency_graph_simple',
    
    ## these are big ones !
    # 'int_superchain_s7_devtooling_repo_eligibility',
    # 'int_superchain_s7_project_to_dependency_graph',
    # 'int_superchain_s7_project_to_project_graph'
]
EXPORT_DIR = 'eval-algos/S7/data'
