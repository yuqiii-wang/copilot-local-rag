import json
import os

def generate_dummy_data():
    confluence_pages = {}
    jira_issues = {}

    topics = [
        {'title': 'Equities Trading Platform', 'desc': 'Documentation for Stock trading. Covers high-frequency trading (HFT) infrastructure, limit orders, market orders, and dark pool integrations.\n\nInput/Output Specs:\n- Input: Order request with symbol, quantity, price, side\n- Output: Execution report with status, filled quantity, average price\n\nExample Trade ID: EQ-2024-00123\n\nISIN Examples: US0378331005 (Apple), US5949181045 (Microsoft)\nSEDOL Examples: 0263494, B0LD5H2\nCUSIP Examples: 037833100 (Apple), 594918104 (Microsoft)\nRIC Examples: AAPL.O (NASDAQ), MSFT.O (NASDAQ)\nFIGI Examples: BBG000B9XRY4 (Apple)\n\nBA Comments: Ensure integration with CBOE and NASDAQ dark pools. Verify HFT latency is below 1ms for market orders.\n\nProcess Flow:\n```mermaid\nsequenceDiagram\n    participant Client as Trading Client\n    participant OMS as Order Management System\n    participant HFT as HFT Engine\n    participant Exchange as Exchange\n    Client->>OMS: Submit Order\n    OMS->>HFT: Route Order\n    HFT->>Exchange: Execute Order\n    Exchange-->>HFT: Execution Confirmation\n    HFT-->>OMS: Execution Report\n    OMS-->>Client: Order Status Update\n```'},
        {'title': 'Fixed Income and Bonds', 'desc': 'System design for municipal, corporate, and government bonds. Features yield-to-maturity calculators, coupon rate tracking, and duration analysis.\n\nInput/Output Specs:\n- Input: Bond ISIN, settlement date, face value\n- Output: Yield-to-maturity, duration, convexity\n\nExample Trade ID: FI-2024-00456\nCUSIP List: 037833100 (Apple), 594918104 (Microsoft)\nAdditional ISIN: US912828R570 (US Treasury)\nSEDOL: 2134234 (US Treasury)\nCUSIP: 912828R57 (US Treasury)\nSOFR Rate Reference: 3.25%\n\nBA Comments: Add support for callable bonds and floating rate notes. Integrate with Bloomberg bond data feeds.\n\nBond Calculation Flow:\n```mermaid\ngraph TD\n    A[Input Bond Data] --> B[Calculate YTM]\n    B --> C[Calculate Duration]\n    C --> D[Calculate Convexity]\n    D --> E[Generate Bond Report]\n    E --> F[Store Results]\n```'},
        {'title': 'Foreign Exchange (FX) Engine', 'desc': 'FX currency pair trading engine. Handles real-time spot rates, forward contracts, currency swaps, and latency-sensitive market making.\n\nInput/Output Specs:\n- Input: Currency pair, amount, trade type, settlement date\n- Output: Execution price, settlement instructions, FX rate\n\nExample Trade ID: FX-2024-00789\nTransaction Code: REF/FX/987654321+00\nCurrency Pairs: EUR/USD, GBP/JPY\nSWIFT Codes: CHASUS33MIA (JP Morgan), DEUTDEFF500 (Deutsche Bank)\n\nBA Comments: Ensure support for all major currency pairs and cross rates. Implement real-time rate feeds from multiple liquidity providers.\n\nFX Trade Flow:\n```mermaid\nsequenceDiagram\n    participant Client as Trading Client\n    participant FxEngine as FX Engine\n    participant Liquidity as Liquidity Providers\n    participant Settlement as Settlement System\n    Client->>FxEngine: FX Trade Request\n    FxEngine->>Liquidity: Get Best Rate\n    Liquidity-->>FxEngine: Rate Quote\n    FxEngine->>FxEngine: Execute Trade\n    FxEngine-->>Client: Trade Confirmation\n    FxEngine->>Settlement: Settlement Instructions\n```'},
        {'title': 'Derivatives and Options', 'desc': 'Architecture for futures, options, and swaps. Includes Black-Scholes pricing models, implied volatility engine, and margin requirement calculators.\n\nInput/Output Specs:\n- Input: Option type, underlying, strike price, expiration date\n- Output: Option price, Greeks (Delta, Gamma, Theta, Vega, Rho)\n\nExample Trade ID: OPT-2024-01011\nOSI Pattern: AAPL240119C00150000\nUnderlying ISIN: US0378331005 (Apple)\nRIC: AAPL.O (NASDAQ)\nFIGI: BBG000B9XRY4 (Apple)\n\nBA Comments: Add support for American and European options. Implement implied volatility surface modeling.\n\nOption Pricing Flow:\n```mermaid\ngraph TD\n    A[Input Option Parameters] --> B[Get Market Data]\n    B --> C[Calculate Implied Volatility]\n    C --> D[Run Black-Scholes Model]\n    D --> E[Calculate Greeks]\n    E --> F[Generate Price Quote]\n```'},
        {'title': 'Risk Management System', 'desc': 'Real-time Value at Risk (VaR) monitoring, stress testing, counterparty credit risk limits, and exposure aggregation across all asset classes.\n\nInput/Output Specs:\n- Input: Portfolio positions, market data, risk parameters\n- Output: VaR report, stress test results, limit breaches\n\nExample Trade ID: RISK-2024-01234\nLEI Target: 5493006MHB84DD0ZWV18\nCounterparty LEI: 549300R0MHBXABCD1234\nISIN: US0378331005 (Apple), US5949181045 (Microsoft)\nSOFR Rate: 3.25%, Fed Rate: 4.50%\n\nBA Comments: Implement scenario-based stress testing. Integrate with market data feeds for real-time risk calculations.\n\nRisk Assessment Flow:\n```mermaid\ngraph TD\n    A[Collect Portfolio Data] --> B[Calculate VaR]\n    B --> C[Run Stress Tests]\n    C --> D[Assess Counterparty Risk]\n    D --> E[Check Risk Limits]\n    E --> F[Generate Risk Report]\n    F --> G[Alert if Breaches]\n```'},
        {'title': 'Order Management System (OMS)', 'desc': 'Core OMS handling routing, execution algos (VWAP, TWAP), FIX protocol endpoints, and broker-dealer order flow.\n\nInput/Output Specs:\n- Input: Order details, routing instructions, algo parameters\n- Output: Order status, execution reports, routing decisions\n\nExample Trade ID: OMS-2024-01567\nISIN: US0378331005 (Apple), US5949181045 (Microsoft)\nLEI: 5493006MHB84DD0ZWV18\nSWIFT: CHASUS33MIA (JP Morgan)\n\nBA Comments: Add support for smart order routing across multiple venues. Implement VWAP and TWAP execution algorithms.\n\nOrder Processing Flow:\n```mermaid\nsequenceDiagram\n    participant Client as Trading Client\n    participant OMS as Order Management System\n    participant Algo as Execution Algorithm\n    participant Venue as Trading Venue\n    Client->>OMS: Submit Order\n    OMS->>OMS: Validate Order\n    OMS->>Algo: Route to Algorithm\n    Algo->>Venue: Execute Order\n    Venue-->>Algo: Execution Reports\n    Algo-->>OMS: Aggregate Results\n    OMS-->>Client: Order Status Update\n```'},
        {'title': 'Ledger and Accounting', 'desc': 'Multi-currency general ledger. Handles daily PnL reconciliation, double-entry bookkeeping, and daily NAV calculations calculation logic.\n\nInput/Output Specs:\n- Input: Trade details, settlement instructions, market prices\n- Output: General ledger entries, PnL reports, NAV statements\n\nExample Trade ID: LEDGER-2024-01890\nISIN: US0378331005 (Apple), US5949181045 (Microsoft)\nFed Rate: 4.50%, SOFR Rate: 3.25%\nFIGI: BBG000B9XRY4 (Apple)\n\nBA Comments: Ensure compliance with IFRS and US GAAP. Implement automated reconciliation with external custodians.\n\nAccounting Flow:\n```mermaid\ngraph TD\n    A[Trade Execution] --> B[Record Journal Entries]\n    B --> C[Daily PnL Calculation]\n    C --> D[Reconcile with Custodians]\n    D --> E[Generate Financial Reports]\n    E --> F[Update NAV]\n```'},
        {'title': 'Market Data Feeds', 'desc': 'Low-latency market data ingestion from Bloomberg, Reuters, and direct exchange feeds (e.g., NASDAQ ITCH, CME MDP).\n\nInput/Output Specs:\n- Input: Market data feed streams, subscription parameters\n- Output: Normalized market data, tick data storage, alerts\n\nExample Trade ID: MD-2024-02123\nRIC Examples: AAPL.O (NASDAQ), MSFT.O (NASDAQ), BT.L (LSE)\nISIN: US0378331005 (Apple), US5949181045 (Microsoft)\n\nBA Comments: Implement redundant feed ingestion for high availability. Add support for historical data replay for backtesting.\n\nMarket Data Flow:\n```mermaid\ngraph TD\n    A[Exchange Feeds] --> B[Feed Handler]\n    C[Bloomberg] --> B\n    D[Reuters] --> B\n    B --> E[Normalization]\n    E --> F[Tick Data Storage]\n    E --> G[Real-time Distribution]\n    G --> H[Strategy Engines]\n```'},
        {'title': 'Algorithmic Trading Execution', 'desc': 'Quantitative model execution environment. Includes backtesting framework, historical tick data storage, and strategy deployment rules.\n\nInput/Output Specs:\n- Input: Strategy parameters, market data, execution rules\n- Output: Trade signals, execution orders, performance metrics\n\nExample Trade ID: ALGO-2024-02456\nISIN: US0378331005 (Apple), US5949181045 (Microsoft)\nRIC: AAPL.O (NASDAQ), MSFT.O (NASDAQ)\nFIGI: BBG000B9XRY4 (Apple)\n\nBA Comments: Add support for machine learning-based strategies. Implement paper trading environment for strategy testing.\n\nAlgorithm Flow:\n```mermaid\nsequenceDiagram\n    participant Data as Market Data\n    participant Model as Quantitative Model\n    participant Signal as Signal Generator\n    participant Execution as Execution Engine\n    Data->>Model: Feed Market Data\n    Model->>Signal: Generate Trading Signals\n    Signal->>Execution: Send Order Instructions\n    Execution->>Execution: Optimize Execution\n    Execution->>Execution: Monitor Performance\n```'},
        {'title': 'Settlement and Clearing', 'desc': 'Post-trade lifecycle. T+1 and T+2 settlement, SWIFT messaging integration, DTC/NSCC clearing, and failed trade resolution workflows.\n\nInput/Output Specs:\n- Input: Trade details, settlement instructions, counterparty info\n- Output: Settlement status, SWIFT messages, exception reports\n\nExample Trade ID: SETTLE-2024-02789\nSwift Pattern: SWT+12345/678\nISIN: US0378331005 (Apple), US912828R570 (US Treasury)\nBuyer LEI: 5493006MHB84DD0ZWV18, Seller LEI: 549300R0MHBXABCD1234\nBuyer SWIFT: CHASUS33MIA (JP Morgan), Seller SWIFT: DEUTDEFF500 (Deutsche Bank)\n\nBA Comments: Ensure compliance with DTCC rules. Implement automated failed trade resolution workflows.\n\nSettlement Flow:\n```mermaid\ngraph TD\n    A[Trade Execution] --> B[Trade Capture]\n    B --> C[Confirmation]\n    C --> D[Matching]\n    D --> E[Clearing]\n    E --> F[Settlement]\n    F --> G[Reconciliation]\n    G --> H[Reporting]\n```'},
        {'title': 'Trade Execution Workflow', 'desc': 'End-to-end trade execution process from order creation to settlement. Includes order_validation, risk_checking, and trade_settlement stages. Last updated: 2023-12-15.\n\nInput/Output Specs:\n- Input: Order request, risk parameters, settlement details\n- Output: Execution status, risk assessment, settlement confirmation\n\nExample Trade ID: WORKFLOW-2024-03011\nISIN: US0378331005 (Apple), US5949181045 (Microsoft)\nLEI: 5493006MHB84DD0ZWV18\nSEDOL: 0263494, CUSIP: 037833100 (Apple)\n\nBA Comments: Map all workflow stages with clear handoffs between systems. Implement real-time status tracking for each trade.\n\nEnd-to-End Workflow:\n```mermaid\ngraph TD\n    A[Order Creation] --> B[Order Validation]\n    B --> C[Risk Checking]\n    C --> D[Order Routing]\n    D --> E[Execution]\n    E --> F[Trade Capture]\n    F --> G[Settlement]\n    G --> H[Reconciliation]\n```'},
        {'title': 'Market Data Processing Pipeline', 'desc': 'Real-time market data processing from feed ingestion to strategy execution. Uses kafka_streams for data processing and redis_cache for low-latency access. Last updated: 2024-01-20.\n\nInput/Output Specs:\n- Input: Raw market data feeds, processing rules\n- Output: Normalized market data, indicators, strategy signals\n\nExample Trade ID: PIPELINE-2024-03234\nSession Num: 123456789\nRIC Examples: AAPL.O (NASDAQ), MSFT.O (NASDAQ), BT.L (LSE)\nISIN: US0378331005 (Apple), US5949181045 (Microsoft)\nFIGI: BBG000B9XRY4 (Apple)\n\nBA Comments: Optimize for low latency processing. Implement fault tolerance and recovery mechanisms.\n\nData Pipeline Flow:\n```mermaid\ngraph TD\n    A[Raw Data Feeds] --> B[Kafka Ingestion]\n    B --> C[Stream Processing]\n    C --> D[Indicator Calculation]\n    D --> E[Redis Cache]\n    E --> F[Strategy Engines]\n    F --> G[Trade Execution]\n```'}
    ]


    # Generate Parent Confluence Page for PROJ
    parent_page_id = "77778881"
    confluence_pages[parent_page_id] = {
        'id': parent_page_id,
        'type': 'page',
        'status': 'current',
        'title': 'PROJ: Project Documentation',
        'version': {
            'number': 1
        },
        'body': {
            'storage': {
                'value': '<div><h1>PROJ Project Documentation</h1><p>Welcome to the PROJ project documentation. This page serves as the parent page for all project-related documentation.</p><ul><li>Overview</li><li>Architecture</li><li>Components</li><li>API Documentation</li><li>Deployment Guide</li></ul></div>',
                'representation': 'storage'
            }
        },
        '_links': {
            'webui': "/spaces/PROJ/pages/77778881/Project+Documentation"
        }
    }

    # Add Feedback Logs Page
    feedback_logs_id = "77778800"
    confluence_pages[feedback_logs_id] = {
        'id': feedback_logs_id,
        'type': 'page',
        'status': 'current',
        'title': 'Feedback Logs',
        'version': {
            'number': 1
        },
        'body': {
            'storage': {
                'value': '<div><h1>Feedback Logs</h1></div>',
                'representation': 'storage'
            }
        },
        '_links': {
            'webui': "/spaces/PROJ/pages/77778800/Feedback+Logs"
        }
    }

    # Jira topic titles and opening sentences used to embed references inside Confluence pages.
    # Must stay in sync with the jira_topics list defined below.
    jira_topic_summaries = [
        ('Fix memory leak in HFT module',            'Trading engine crashes under high load — profiling and memory allocation fixes required'),
        ('Upgrade FIX protocol library',             'Update QuickFIX engine to version 1.15 to resolve latency spike bugs'),
        ('Add OAuth2 to oms API',                    'Secure the Order Management System backend endpoints with OAuth2 authentication'),
        ('Refactor PnL reconciliation job',          'Daily PnL job is timing out — SQL queries and indexing need optimisation'),
        ('Implement Black-Scholes Greeks',           'Add Delta, Gamma, Theta, Vega, and Rho calculations to the options pricing engine'),
        ('Create Dockerfile for FX Engine',          'Containerise the Foreign Exchange engine for Kubernetes deployment'),
        ('Add Prometheus metrics to ledger',         'Expose /metrics endpoint in the Ledger service with transaction counts and error rates'),
        ('Fix UI sorting in Bond Yield table',       'The frontend yield-to-maturity table currently sorts numbers alphabetically instead of numerically'),
        ('Migrate market data to Kafka',             'Move from RabbitMQ to Kafka for Bloomberg feed ingestion to meet throughput requirements'),
        ('Audit SWIFT messaging integration',        'Perform security audit and compliance checks on all SWIFT post-trade message flows'),
        ('Implement order validation service',       'Create a new microservice for order validation integrating with the risk management system'),
        ('Optimize market data processing',          'Improve market data processing pipeline performance by implementing parallel stream processing'),
    ]

    # Map confluence pages to related jira issues
    # Maps topic index (0-based) to list of jira issue indices (1-based)
    confluence_to_jira_mapping = {
        0: [1, 2],           # Equities Trading Platform → HFT memory leak, FIX protocol upgrade
        1: [5, 9],           # Fixed Income and Bonds → Black-Scholes Greeks, UI sorting fix
        2: [7],              # FX Engine → Kafka migration
        3: [5],              # Derivatives and Options → Black-Scholes Greeks
        4: [4],              # Risk Management System → PnL reconciliation
        5: [3, 11],          # Order Management System → OAuth2, order validation service
        6: [4, 8],           # Ledger and Accounting → PnL reconciliation, Prometheus metrics
        7: [10, 12],         # Market Data Feeds → SWIFT audit, market data optimization
        8: [12],             # Algorithmic Trading Execution → market data optimization
        9: [10],             # Settlement and Clearing → SWIFT audit
        10: [11],            # Trade Execution Workflow → order validation service
        11: [12]             # Market Data Processing Pipeline → market data optimization
    }

    # Generate Confluence Pages
    for i, topic in enumerate(topics, 1):
        page_id = str(77778881 + i)
        title = topic['title']
        desc = topic['desc']
        
        # Get related jira issues for this page
        related_jira_indices = confluence_to_jira_mapping.get(i - 1, [])
        jira_references = ''
        if related_jira_indices:
            sentences = []
            for jira_idx in related_jira_indices:
                jira_key = f'PROJ-{jira_idx}'
                jira_title, first_sentence = jira_topic_summaries[jira_idx - 1]
                sentences.append(f'<strong>{jira_key}</strong>: {jira_title} — {first_sentence}.')
            jira_references = ('<h3>Related Jira Issues</h3><p>'
                               'The following Jira tickets track the implementation work for this component. '
                               + ' '.join(sentences) + '</p>')
        
        confluence_pages[page_id] = {
            'id': page_id,
            'type': 'page',
            'status': 'current',
            'title': f'PROJ: {title}',
            'version': {
                'number': 1
            },
            'body': {
                'storage': {
                    'value': f'<div><h1>{title}</h1><p>{desc}</p>{jira_references}<ul><li>Architecture Overview</li><li>API Endpoints</li></ul></div>',
                    'representation': 'storage'
                }
            },
            '_links': {
                'webui': f"/spaces/PROJ/pages/{page_id}/{title.replace(' ', '+')}"
            }
        }

    jira_topics = [
        {'title': 'Fix memory leak in HFT module', 'desc': 'Trading engine crashes under high load. Profile memory and fix the leak.\n\nInput/Output Specs:\n- Input: Load test parameters, memory profiling data\n- Output: Fixed code, memory usage report, performance metrics\n\nExample Trade ID: HFT-2024-00123\n\nBA Comments: The memory leak occurs during peak trading hours when handling >10,000 orders per second. Prioritize fixing this for the Q3 release.\n\nMemory Leak Analysis Flow:\n```mermaid\ngraph TD\n    A[Run Load Tests] --> B[Profile Memory Usage]\n    B --> C[Identify Leak Source]\n    C --> D[Fix Memory Allocation]\n    D --> E[Verify Fix with Load Tests]\n    E --> F[Document Changes]\n```'},
        {'title': 'Upgrade FIX protocol library', 'desc': 'Update QuickFIX engine to version 1.15 to resolve latency spike bugs.\n\nInput/Output Specs:\n- Input: Current FIX messages, test scenarios\n- Output: Updated library, regression test results, latency measurements\n\nExample Trade ID: FIX-2024-00456\n\nBA Comments: Ensure backward compatibility with existing FIX sessions. Test with real market data to verify latency improvements.\n\nUpgrade Process:\n```mermaid\nsequenceDiagram\n    participant Dev as Development\n    participant Test as Testing\n    participant Prod as Production\n    Dev->>Dev: Update QuickFIX Library\n    Dev->>Test: Deploy to Test Environment\n    Test->>Test: Run Regression Tests\n    Test->>Test: Measure Latency\n    Test-->>Dev: Test Results\n    Dev->>Prod: Deploy to Production\n    Prod->>Prod: Monitor Performance\n```'},
        {'title': 'Add OAuth2 to oms API', 'desc': 'Secure the Order Management System backend endpoints with OAuth2.\n\nInput/Output Specs:\n- Input: API endpoints, authentication requirements\n- Output: OAuth2 implementation, security test results, documentation\n\nExample Trade ID: OAUTH-2024-00789\n\nBA Comments: Integrate with existing identity provider. Implement role-based access control for different user types.\n\nAuthentication Flow:\n```mermaid\nsequenceDiagram\n    participant Client as Client\n    participant API as OMS API\n    participant OAuth as OAuth Server\n    Client->>OAuth: Request Token\n    OAuth-->>Client: Access Token\n    Client->>API: Request with Token\n    API->>OAuth: Validate Token\n    OAuth-->>API: Token Valid\n    API-->>Client: Authorized Response\n```'},
        {'title': 'Refactor PnL reconciliation job', 'desc': 'Daily PnL job is timing out. Optimize the SQL queries and add indexing.\n\nInput/Output Specs:\n- Input: Current SQL queries, database schema\n- Output: Optimized queries, indexed schema, performance report\n\nExample Trade ID: PNL-2024-01011\n\nBA Comments: The job currently takes >4 hours to run. Target performance is <30 minutes. Focus on the largest tables first.\n\nOptimization Flow:\n```mermaid\ngraph TD\n    A[Analyze Current Queries] --> B[Identify Bottlenecks]\n    B --> C[Add Indexes]\n    C --> D[Rewrite Queries]\n    D --> E[Test Performance]\n    E --> F[Deploy to Production]\n    F --> G[Monitor Execution Time]\n```'},
        {'title': 'Implement Black-Scholes Greeks', 'desc': 'Add Delta, Gamma, Theta, Vega, and Rho calculations to the options engine.\n\nInput/Output Specs:\n- Input: Option parameters, market data\n- Output: Greek values, sensitivity analysis, validation results\n\nExample Trade ID: GREEKS-2024-01234\n\nBA Comments: Ensure calculations are accurate to 6 decimal places. Implement caching for frequently requested values.\n\nGreeks Calculation Flow:\n```mermaid\ngraph TD\n    A[Input Option Data] --> B[Calculate Delta]\n    A --> C[Calculate Gamma]\n    A --> D[Calculate Theta]\n    A --> E[Calculate Vega]\n    A --> F[Calculate Rho]\n    B --> G[Generate Sensitivity Report]\n    C --> G\n    D --> G\n    E --> G\n    F --> G\n```'},
        {'title': 'Create Dockerfile for FX Engine', 'desc': 'Containerize the Foreign Exchange engine for Kubernetes deployment.\n\nInput/Output Specs:\n- Input: Application dependencies, configuration requirements\n- Output: Dockerfile, Kubernetes manifests, deployment documentation\n\nExample Trade ID: DOCKER-2024-01567\n\nBA Comments: Ensure the container image is lightweight. Implement health checks and liveness probes for Kubernetes.\n\nContainerization Flow:\n```mermaid\ngraph TD\n    A[Create Dockerfile] --> B[Build Image]\n    B --> C[Test Locally]\n    C --> D[Push to Registry]\n    D --> E[Create Kubernetes Manifests]\n    E --> F[Deploy to Cluster]\n    F --> G[Verify Deployment]\n```'},
        {'title': 'Add Prometheus metrics to ledger', 'desc': 'Expose /metrics endpoint in Ledger service with transaction counts and errors.\n\nInput/Output Specs:\n- Input: Service endpoints, metric requirements\n- Output: Prometheus integration, dashboard configurations, alert rules\n\nExample Trade ID: METRICS-2024-01890\n\nBA Comments: Include metrics for transaction volume, processing time, and error rates. Set up alerts for异常 patterns.\n\nMetrics Flow:\n```mermaid\ngraph TD\n    A[Instrument Code] --> B[Expose Metrics Endpoint]\n    B --> C[Configure Prometheus]\n    C --> D[Create Grafana Dashboards]\n    D --> E[Set Up Alert Rules]\n    E --> F[Monitor Metrics]\n```'},
        {'title': 'Fix UI sorting in Bond Yield table', 'desc': 'The frontend yield-to-maturity table sorts numbers alphabetically. Fix this.\n\nInput/Output Specs:\n- Input: Current table implementation, test data\n- Output: Fixed sorting logic, test cases, UI verification\n\nExample Trade ID: UI-2024-02123\n\nBA Comments: Ensure the fix works for both ascending and descending sort directions. Test with various yield values including negative yields.\n\nSorting Fix Flow:\n```mermaid\ngraph TD\n    A[Identify Sorting Issue] --> B[Analyze Current Code]\n    B --> C[Implement Numeric Sorting]\n    C --> D[Test with Test Data]\n    D --> E[Verify UI Behavior]\n    E --> F[Document Fix]\n```'},
        {'title': 'Migrate market data to Kafka', 'desc': 'Move from RabbitMQ to Kafka for Bloomberg feed ingestion due to throughput.\n\nInput/Output Specs:\n- Input: Current RabbitMQ configuration, feed specifications\n- Output: Kafka implementation, performance test results, migration plan\n\nExample Trade ID: KAFKA-2024-02456\n\nBA Comments: Implement a dual-write strategy during migration to ensure no data loss. Test with peak market data volumes.\n\nMigration Flow:\n```mermaid\nsequenceDiagram\n    participant Source as RabbitMQ\n    participant Dual as Dual-Write Adapter\n    participant Target as Kafka\n    participant Apps as Consumer Apps\n    Source->>Dual: Message\n    Dual->>Source: Acknowledge\n    Dual->>Target: Replicate Message\n    Target->>Apps: Consume Messages\n    Apps-->>Target: Acknowledge\n```'},
        {'title': 'Audit SWIFT messaging integration', 'desc': 'Perform security audit and compliance checks on the SWIFT post-trade messages.\n\nInput/Output Specs:\n- Input: SWIFT message flows, compliance requirements\n- Output: Audit report, security recommendations, compliance checklist\n\nExample Trade ID: SWIFT-2024-02789\n\nBA Comments: Ensure compliance with SWIFT CSP standards. Review all message types used in post-trade processing.\n\nAudit Process:\n```mermaid\ngraph TD\n    A[Review SWIFT Flows] --> B[Check Security Controls]\n    B --> C[Verify Compliance]\n    C --> D[Identify Vulnerabilities]\n    D --> E[Generate Audit Report]\n    E --> F[Implement Recommendations]\n```'},
        {'title': 'Implement order validation service', 'desc': 'Create a new microservice for order validation that integrates with risk management system. Use validateOrderRequest for request validation. Due date: 2024-02-15.\n\nInput/Output Specs:\n- Input: Order data, risk parameters\n- Output: Validation results, risk assessment, error codes\n\nExample Trade ID: VALIDATION-2024-03011\n\nBA Comments: The service must process 1000+ orders per second. Integrate with the existing risk management system via REST API.\n\nValidation Flow:\n```mermaid\nsequenceDiagram\n    participant OMS as Order Management\n    participant Validation as Validation Service\n    participant Risk as Risk Management\n    OMS->>Validation: Order Validation Request\n    Validation->>Risk: Risk Check\n    Risk-->>Validation: Risk Assessment\n    Validation->>Validation: Validate Order\n    Validation-->>OMS: Validation Result\n```'},
        {'title': 'Optimize market data processing', 'desc': 'Improve market data processing pipeline performance by implementing parallel processing. Use kafka_streams for real-time processing. Due date: 2024-03-10.\n\nInput/Output Specs:\n- Input: Raw market data, processing rules\n- Output: Processed data, performance metrics, scaling analysis\n\nExample Trade ID: OPTIMIZE-2024-03234\n\nBA Comments: Target 50% reduction in processing latency. Implement backpressure mechanisms to handle peak data rates.\n\nOptimization Flow:\n```mermaid\ngraph TD\n    A[Analyze Current Pipeline] --> B[Identify Bottlenecks]\n    B --> C[Implement Parallel Processing]\n    C --> D[Configure Kafka Streams]\n    D --> E[Test Performance]\n    E --> F[Deploy to Production]\n    F --> G[Monitor Metrics]\n```'}
    ]

    # Generate Jira Issues
    for i, topic in enumerate(jira_topics, 1):
        issue_id = str(10000 + i)
        issue_key = f'PROJ-{i}'
        title = topic['title']
        desc = topic['desc']
        
        # Generate trade IDs and example data based on issue type
        trade_id = f'TRADE-{100000 + i:06d}'
        ngrams = ['high-frequency', 'market-data', 'order-execution', 'risk-management', 'trade-settlement']
        
        # Create enriched description with examples
        # Generate JSON examples
        input_json = {
            "tradeId": trade_id,
            "symbol": 'AAPL' if i % 3 == 0 else 'MSFT' if i % 3 == 1 else 'GOOGL',
            "quantity": 100 * (i + 1),
            "price": round(100 + (i * 10.5), 2),
            "side": 'BUY' if i % 2 == 0 else 'SELL'
        }
        
        output_json = {
            "status": 'EXECUTED' if i % 2 == 0 else 'PENDING',
            "executionId": f"EXEC-{10000 + i:05d}",
            "timestamp": f"2024-03-{10 + i % 20:02d}T14:{30 + i % 30:02d}:{i % 60:02d}Z",
            "filledQuantity": 100 * (i + 1) if i % 2 == 0 else 0,
            "averagePrice": round(100 + (i * 10.5), 2) if i % 2 == 0 else None
        }
        
        # Convert to JSON strings
        import json
        input_json_str = json.dumps(input_json, indent=2)
        output_json_str = json.dumps(output_json, indent=2)
        
        enriched_desc = f'''<div>
            <h3>Task: {title}</h3>
            <p>{desc}</p>
            
            <h4>Technical Details</h4>
            <ul>
                <li>Trade ID: {trade_id}</li>
                <li>Related N-grams: {', '.join(ngrams[:3 + i % 3])}</li>
                <li>Priority: {'High' if i % 2 == 0 else 'Medium'}</li>
                <li>Estimated Effort: {3 + i % 5} days</li>
            </ul>
            
            <h4>Example Input/Output</h4>
            <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
                <strong>Input:</strong>
                <pre>{input_json_str}</pre>
                <strong>Output:</strong>
                <pre>{output_json_str}</pre>
            </div>
            
            <h4>Implementation Notes</h4>
            <p>Requires urgent implementation for the Q3 release. Must integrate with existing {ngrams[0]} module and follow {ngrams[1]} best practices. Test with realistic market data scenarios including high-volume trading periods.</p>
            
            <p>Additional context: This task is part of the {['HFT Optimization', 'Risk Management Overhaul', 'Market Data Pipeline Upgrade', 'Order Execution Enhancement'][i % 4]} initiative.</p>
        </div>'''
        
        jira_issues[issue_key] = {
            'id': issue_id,
            'key': issue_key,
            'fields': {
                'summary': f'Implement {title}',
                'description': enriched_desc
            }
        }

    data = {
        'confluence': confluence_pages,
        'jira': jira_issues
    }

    output_path = os.path.join(os.path.dirname(__file__), 'dummy_data.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f'Generated dummy data successfully to {output_path}')

if __name__ == '__main__':
    generate_dummy_data()

