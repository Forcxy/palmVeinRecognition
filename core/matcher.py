import numpy as np
from typing import List, Tuple, Dict

class FeatureMatcher:
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def match(self, query_vector: np.ndarray, database: Dict[str, np.ndarray], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        在数据库中匹配最相似的特征
        返回: 排序后的匹配结果列表[(name, similarity), ...]
        """
        if not database:
            return []
        
        similarity_scores = []
        for name, db_vec in database.items():
            sim = self.cosine_similarity(query_vector, db_vec)
            similarity_scores.append((name, sim))
        
        # 按相似度降序排序
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        return similarity_scores[:top_k]