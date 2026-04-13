COMPANY_CONTACT_CASES = [
    {
        "id": "company-address-where",
        "question": "cong ty o dau",
        "expected_keywords": ["98/12 nguyen xi", "binh thanh"],
        "forbidden_keywords": ["0909.268.416", "info@nvdkarchitects.vn", "thu 2 - thu 7"],
    },
    {
        "id": "company-address-place",
        "question": "cong ty o cho nao",
        "expected_keywords": ["98/12 nguyen xi", "binh thanh"],
        "forbidden_keywords": ["0909.268.416", "info@nvdkarchitects.vn", "thu 2 - thu 7"],
    },
    {
        "id": "company-address-street",
        "question": "cong ty o duong nao",
        "expected_keywords": ["98/12 nguyen xi", "binh thanh"],
        "forbidden_keywords": ["0909.268.416", "info@nvdkarchitects.vn", "thu 2 - thu 7"],
    },
    {
        "id": "company-hotline",
        "question": "hotline la gi",
        "expected_keywords": ["0909.268.416"],
        "forbidden_keywords": ["98/12 nguyen xi", "info@nvdkarchitects.vn", "thu 2 - thu 7"],
    },
    {
        "id": "company-email",
        "question": "email la gi",
        "expected_keywords": ["info@nvdkarchitects.vn"],
        "forbidden_keywords": ["98/12 nguyen xi", "0909.268.416", "thu 2 - thu 7"],
    },
    {
        "id": "company-working-hours",
        "question": "gio lam viec the nao",
        "expected_keywords": ["thu 2 - thu 7", "8:00 - 18:00"],
        "forbidden_keywords": ["98/12 nguyen xi", "0909.268.416", "info@nvdkarchitects.vn"],
    },
    {
        "id": "company-overview",
        "question": "gioi thieu cong ty",
        "expected_keywords": [
            "thong tin ve nguyen vo dang khoa architects",
            "gioi thieu ngan:",
            "dia chi:",
            "hotline:",
            "email:",
            "gio lam viec:",
        ],
        "forbidden_keywords": [],
    },
]
