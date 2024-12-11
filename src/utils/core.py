
FULL_HACKS_CLASSIFICATIONS_DICT = {
    "Popularity": {
        "description": "How widely known or utilized the hack is.",
        "categories": [
            {"name": "Widely Known", "description": "Hacks that are commonly recognized and frequently used by the general public."},
            {"name": "Moderately Known", "description": "Hacks that are familiar within certain groups or require some research to discover."},
            {"name": "Little Known", "description": "Hacks that are rare, niche, or not widely publicized."},
            {"name": "Unknown Popularity", "description": "When the popularity of the hack is unclear or not easily categorized."},
        ],
        "type": "single_cat",
    },
    "Specific Financial Topic": {
        "description": "Specific financial areas addressed by the hack.",
        "categories": [
            {"name": "Banking and Financial Services", "description": "Hacks related to bank accounts, fees, and banking services."},
            {"name": "Credit Cards", "description": "Strategies for managing and optimizing credit card usage."},
            {"name": "Loans and Debt Management", "description": "Hacks aimed at managing, reducing, or refinancing loans and debts."},
            {"name": "Real Estate and Housing", "description": "Hacks for buying, selling, or investing in property."},
            {"name": "Budgeting and Saving", "description": "Methods to save money and manage personal budgets."},
            {"name": "Tax Strategies", "description": "Hacks to legally reduce tax liabilities."},
            {"name": "Retirement Planning", "description": "Tips for securing financial stability in retirement."},
            {"name": "Insurance", "description": "Strategies related to various insurance products."},
            {"name": "Investments", "description": "Hacks focused on investing strategies to grow wealth."},
            {"name": "Education Funding", "description": "Hacks for saving on education costs."},
            {"name": "No Specific Topic", "description": "Hacks that do not fit into any particular financial topic."},
        ],
        "type": "multi_cat",
    },
    "Audience and Life Stage": {
        "description": "Primary audience that would benefit from the hack.",
        "categories": [
            {"name": "Students and Young Adults", "description": "Hacks suited for individuals in their educational or early career stages."},
            {"name": "Families", "description": "Hacks designed to help households manage collective finances."},
            {"name": "Professionals and Entrepreneurs", "description": "Hacks aimed at individuals working in professional roles or running businesses."},
            {"name": "Retirees and Seniors", "description": "Hacks relevant for financial management in retirement."},
            {"name": "All Audiences", "description": "Hacks applicable to a broad range of individuals, regardless of age or life stage."},
        ],
        "type": "multi_cat",
    },
    "Time Horizon": {
        "description": 'How quickly the benefits of the hack are realized.',
        "categories": [
            {"name": 'Short Term', "description": 'Hacks that provide immediate or near-immediate financial gains.' },
            {"name": 'Medium Term', "description": 'Hacks that yield benefits over a moderate period, typically within a year.' },
            {"name": 'Long Term', "description": 'Hacks that offer financial advantages over an extended period, often several years.' },
            {"name": 'Unspecified Time Horizon', "description": 'When the time frame for benefits is unclear.' }
        ],
        "type": 'single_cat'
    },
    "Risk Level": {
        "description": 'Financial risk associated with implementing the hack.',
        "categories": [
            {"name": 'Low Risk', "description": 'Hacks that involve minimal financial risk.' },
            {"name": 'Moderate Risk', "description": 'Hacks that carry a balanced level of risk and potential reward.' },
            {"name": 'High Risk', "description": 'Hacks that involve significant financial risk and potential for loss.' },
            {"name": 'Unknown Risk Level', "description": 'When the risk associated with the hack is unclear.' }
        ],
        "type": 'single_cat'
    },
    "Implementation Difficulty": {
        "description": 'Time, resources, and complexity involved in implementing the hack.',
        "categories": [
            {"name": 'Minimal Effort', "description": 'Hacks that can be implemented quickly with little to no additional resources.' },
            {"name": 'Moderate Effort', "description": 'Hacks that require a moderate amount of time, resources, or planning.' },
            {"name": 'Significant Effort', "description": 'Hacks that demand substantial time, resources, or complex planning.' },
            {"name": 'Unknown Implementation Effort', "description": 'When the level of effort required is unclear.' }
        ],
        "type": 'single_cat'
    },
    "Financial Goals": {
        "description": 'Financial objectives the hack helps achieve.',
        "categories": [
            {"name": 'Debt Reduction', "description": 'Hacks aimed at minimizing or eliminating debt.' },
            {"name": 'Wealth Building', "description": 'Hacks focused on increasing overall financial wealth.' },
            {"name": 'Income Generation', "description": 'Hacks designed to create additional streams of income.' },
            {"name": 'Expense Management', "description": 'Hacks that help control or reduce expenses.' },
            {"name": 'Other Financial Goals', "description": 'Hacks that align with financial objectives not explicitly listed.' }
        ],
        "type": 'multi_cat'
    },
    "Knowledge Level Required": {
        "description": "User's understanding of finance needed for the hack.",
        "categories": [
            {"name": 'Beginner', "description": 'Hacks that require basic financial knowledge.' },
            {"name": 'Intermediate', "description": 'Hacks that require a moderate understanding of financial concepts.' },
            {"name": 'Advanced', "description": 'Hacks that necessitate a deep or sophisticated understanding of finance.' },
            {"name": 'Unknown Literacy Level', "description": 'When the required financial knowledge for the hack is unclear.' }
        ],
        "type": 'single_cat'
    }
}

HACKS_CLASSIFICATIONS_CATEGORIES = {
    "Popularity": { "categories": ["Widely Known", "Moderately Known", "Little Known", "Unknown Popularity"], "exclude": "Unknown Popularity" },
    "Specific Financial Topic": { "categories": ["Banking and Financial Services", "Credit Cards", "Loans and Debt Management", "Real Estate and Housing", "Budgeting and Saving", 
                                    "Tax Strategies", "Retirement Planning", "Insurance", "Investments", "Education Funding", "No Specific Topic"], "exclude": "No Specific Topic" },
    "Audience and Life Stage": { "categories": ["Students and Young Adults", "Families", "Professionals and Entrepreneurs", "Retirees and Seniors", "All Audiences"] },
    "Time Horizon": { "categories": ['Short Term', 'Medium Term', 'Long Term', 'Unspecified Time Horizon'], "exclude": "Unspecified Time Horizon" },
    "Risk Level": { "categories": ['Low Risk', 'Moderate Risk', 'High Risk', 'Unknown Risk Level' ],  "exclude": "Unknown Risk Level" },
    "Implementation Difficulty": { "categories": ['Minimal Effort', 'Moderate Effort', 'Significant Effort', 'Unknown Implementation Effort'], "exclude": "Unknown Implementation Effort" },
    "Financial Goals": { "categories": ['Debt Reduction', 'Wealth Building', 'Income Generation', 'Expense Management', 'Other Financial Goals' ], "exclude": "Other Financial Goals" },
    "Knowledge Level Required": { "categories": ['Beginner', 'Intermediate', 'Advanced', 'Unknown Literacy Level'], "exclude": "Unknown Literacy Level" }
}