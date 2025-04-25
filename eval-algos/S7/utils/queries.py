"""
This module contains all the SQL queries used to fetch data from OSO.
"""

QUERIES = [
    {
        "filename": "onchain__project_metadata",
        "filetype": "csv",
        "query": """
            SELECT
                p.project_id,
                p.project_name,
                p.display_name,
                e.transaction_count,
                e.gas_fees,
                e.active_days,
                e.meets_all_criteria AS is_eligible
            FROM int_superchain_s7_onchain_builder_eligibility AS e
            JOIN projects_v1 AS p ON e.project_id = p.project_id
            JOIN projects_by_collection_v1 AS pbc ON p.project_id = pbc.project_id
            WHERE
                pbc.collection_name = '8-2'
                AND e.sample_date = DATE '2025-03-01'
            ORDER BY e.transaction_count DESC
        """
    },
    {
        "filename": "onchain__metrics_by_project",
        "filetype": "csv",
        "query": """
            SELECT
                m.project_id,
                p.display_name,
                pbc.project_name,
                m.chain,
                m.metric_name,
                DATE_FORMAT(m.sample_date, '%Y-%m-%d') AS sample_date,
                DATE_FORMAT(m.sample_date, '%b %Y') AS measurement_period,
                m.amount
            FROM int_superchain_s7_onchain_metrics_by_project AS m
            JOIN projects_by_collection_v1 AS pbc ON m.project_id = pbc.project_id
            JOIN projects_v1 AS p ON pbc.project_id = p.project_id
            WHERE
                pbc.collection_name = '8-2'
                AND m.sample_date >= DATE '2025-02-01'
                AND m.sample_date < DATE '2025-04-01'
        """
    },
    {
        "filename": "devtooling__project_metadata",
        "filetype": "csv",
        "query": """
            SELECT 
                project_id,
                project_name,
                display_name,
                fork_count,
                star_count,
                num_packages_in_deps_dev
            FROM int_superchain_s7_devtooling_metrics_by_project
            ORDER BY fork_count DESC
        """
    },
    {
        "filename": "devtooling__onchain_metadata",
        "filetype": "csv",
        "query": """
            SELECT DISTINCT
                b.project_id,
                p.project_name,
                p.display_name,
                MAX(b.total_transaction_count) AS total_transaction_count,
                MAX(b.total_gas_fees) AS total_gas_fees
            FROM int_superchain_s7_devtooling_onchain_builder_nodes AS b
            JOIN projects_v1 AS p ON b.project_id = p.project_id
            GROUP BY 1, 2, 3
            ORDER BY 4 DESC
        """
    },
    {
        "filename": "devtooling__dependency_graph",
        "filetype": "csv",
        "query": """
            SELECT *
            FROM int_superchain_s7_devtooling_deps_to_projects_graph
        """
    },
    {
        "filename": "devtooling__developer_graph",
        "filetype": "csv",
        "query": """
            SELECT *
            FROM int_superchain_s7_devtooling_devs_to_projects_graph
        """
    },
    {
        "filename": "devtooling__raw_metrics",
        "filetype": "json",
        "query": """
            SELECT * 
            FROM int_superchain_s7_devtooling_metrics_by_project
        """
    },
] 