PROJECT_ANSWER_REGRESSION_CASES = [
    {
        "id": "project-single-suggestion",
        "question": "cho toi 1 du an",
        "expected_keywords": ["Mình gợi ý dự án này", "Nhà vườn Bình Tân", "Bình Tân"],
        "forbidden_keywords": ["Mình chưa có câu trả lời grounded", "Khách sạn & Resort"],
    },
    {
        "id": "project-company-list",
        "question": "du an cua cong ty ban la gi",
        "expected_keywords": ["Mình tìm thấy một vài dự án nổi bật", "Nhà vườn Bình Tân", "Resort Hải Vân 2"],
        "forbidden_keywords": ["Mình chưa có câu trả lời grounded"],
    },
    {
        "id": "project-location-list",
        "question": "du an nao o Binh Tan",
        "expected_keywords": ["Bình Tân"],
        "forbidden_keywords": ["Resort Hải Vân 2"],
    },
]
