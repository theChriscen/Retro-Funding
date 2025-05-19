"""
This module contains all the SQL queries used to fetch data from OSO.
"""

THIS_DEVTOOLING_COLLECTION = '8-3'
SAMPLE_DATE = '2025-04-01'
START_DATE = '2025-03-01'
END_DATE = '2025-05-01'

QUERIES = [
    {
        "filename": "onchain__project_metadata",
        "filetype": "csv",
        "query": f"""
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
                pbc.collection_name = '{THIS_DEVTOOLING_COLLECTION}'
                AND e.sample_date = DATE '{SAMPLE_DATE}'
            ORDER BY e.transaction_count DESC
        """
    },
    {
        "filename": "onchain__metrics_by_project",
        "filetype": "csv",
        "query": f"""
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
                pbc.collection_name = '{THIS_DEVTOOLING_COLLECTION}'
                AND m.sample_date >= DATE ('{START_DATE}')
                AND m.sample_date < DATE '{END_DATE}'
                AND NOT ( -- TODO: needs to be removed on OP Atlas side
                  p.project_name = '0xeb6d215732b1ed881e718faa8bf8b4b88a94edf021efa9a3cf3e2cc3c22b0961'
                  AND m.metric_name = 'average_tvl_monthly'
                )
        """
    },
    {
        "filename": "devtooling__project_metadata",
        "filetype": "csv",
        "query": f"""
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
        "query": f"""
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
        "query": f"""
            SELECT *
            FROM int_superchain_s7_devtooling_deps_to_projects_graph
        """
    },
    {
        "filename": "devtooling__developer_graph",
        "filetype": "csv",
        "query": f"""
            SELECT *
            FROM int_superchain_s7_devtooling_devs_to_projects_graph
        """
    },
    {
        "filename": "devtooling__raw_metrics",
        "filetype": "json",
        "query": f"""
            SELECT * 
            FROM int_superchain_s7_devtooling_metrics_by_project
        """
    },
    {
        "filename": "onchain__summary_metric_snapshot",
        "filetype": "csv",
        "query": f"""
            WITH params AS (
                SELECT DATE '{SAMPLE_DATE}' AS month_start
            )
            SELECT
                tm.project_id,
                p.project_name AS op_atlas_id,
                p.display_name,
                LOWER(
                    REGEXP_REPLACE(m.display_name, '[^a-zA-Z0-9]+', '_')
                ) || '__' ||
                LOWER(DATE_FORMAT(tm.sample_date, '%b')) || '_' || 
                DATE_FORMAT(tm.sample_date, '%Y')
                AS metric_name,

                DATE_FORMAT(tm.sample_date, '%b %Y') AS measurement_period,

                SUM(
                    CASE
                        WHEN m.display_name IN ('Defillama TVL', 'Active Addresses Aggregation')
                        THEN tm.amount / DAY(LAST_DAY_OF_MONTH(tm.sample_date))
                        ELSE tm.amount
                    END
                ) AS amount
            FROM timeseries_metrics_by_project_v0 AS tm
            JOIN metrics_v0 AS m ON m.metric_id = tm.metric_id
            JOIN projects_v1 AS p ON p.project_id = tm.project_id
            JOIN projects_by_collection_v1 AS pbc ON pbc.project_id = p.project_id
            JOIN params AS pms ON TRUE
            WHERE
                pbc.collection_name = '{THIS_DEVTOOLING_COLLECTION}'
            AND tm.sample_date >= pms.month_start
            AND tm.sample_date < DATE_ADD('month', 1, pms.month_start)
            AND (
                    m.metric_name LIKE '%gas_fees_daily'
                OR m.metric_name LIKE '%defillama_tvl_daily'
                OR m.metric_name LIKE '%active_addresses_aggregation_daily'
                OR m.metric_name LIKE '%contract_invocations_daily'
            )
            GROUP BY 1, 2, 3, 4, 5
        """
    }
] 