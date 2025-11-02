"""
RAG Evaluation Script for PromptLayer
Comprehensive evaluation of the Moraqib RAG system with detailed metrics
"""

import os
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if PROMPTLAYER_API_KEY is set
if not os.getenv("PROMPTLAYER_API_KEY"):
    print("⚠️ Warning: PROMPTLAYER_API_KEY not found in environment")
    print("   Please set it in your .env file or environment variables")

from moraqib_rag import MoraqibRAG
from firestore_handler import firestore_handler

class RAGEvaluator:
    def __init__(self):
        """Initialize RAG evaluator with test queries and metrics"""
        self.rag = MoraqibRAG()
        self.evaluation_results = []
        
        # Test query categories for comprehensive evaluation
        self.test_queries = {
            "basic_queries": [
                "How many detections do we have?",
                "Show me all reports",
                "What detections happened today?",
                "Give me a summary of recent activity"
            ],
            "time_filtered_queries": [
                "How many detections yesterday?",
                "Show me detections from last week",
                "What happened last night?",
                "Any detections in the last hour?"
            ],
            "specific_search_queries": [
                "Show me woodland detections",
                "Find camouflage reports",
                "Any desert environment detections?",
                "Show me reports with equipment"
            ],
            "analytical_queries": [
                "Which device has the most detections?",
                "What's the average soldier count?",
                "Show me high-confidence detections",
                "What environments are most common?"
            ],
            "edge_cases": [
                "Tell me about the weather",
                "What's 2+2?",
                "Show me reports from Mars",
                "How do I cook pasta?"
            ]
        }
    
    async def run_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive RAG evaluation
        
        Returns:
            Dictionary with evaluation results and metrics
        """
        print("🔬 Starting RAG Evaluation with PromptLayer...")
        print("=" * 60)
        
        # Initialize Firestore if needed
        if not firestore_handler.db:
            firestore_handler.initialize()
        
        total_queries = sum(len(queries) for queries in self.test_queries.values())
        current_query = 0
        
        evaluation_start = time.time()
        
        for category, queries in self.test_queries.items():
            print(f"\n📊 Testing {category.replace('_', ' ').title()}...")
            
            for query in queries:
                current_query += 1
                print(f"\n[{current_query}/{total_queries}] Query: {query}")
                
                try:
                    # Run query and measure performance
                    query_start = time.time()
                    result = await self.rag.query(query)
                    query_time = time.time() - query_start
                    
                    # Analyze result quality
                    quality_score = self._evaluate_response_quality(query, result)
                    
                    # Store evaluation result
                    eval_result = {
                        "query": query,
                        "category": category,
                        "success": result.get("success", False),
                        "response_time": query_time,
                        "quality_score": quality_score,
                        "reports_used": result.get("reports_count", 0),
                        "answer_length": len(result.get("answer", "")),
                        "timestamp": datetime.now().isoformat(),
                        "answer": result.get("answer", "")[:200] + "..." if len(result.get("answer", "")) > 200 else result.get("answer", "")
                    }
                    
                    self.evaluation_results.append(eval_result)
                    
                    # Print result summary
                    status = "✅" if result.get("success") else "❌"
                    print(f"   {status} Success: {result.get('success')}")
                    print(f"   ⏱️  Time: {query_time:.2f}s")
                    print(f"   📊 Quality: {quality_score}/10")
                    print(f"   📚 Reports: {result.get('reports_count', 0)}")
                    
                except Exception as e:
                    print(f"   ❌ Error: {str(e)}")
                    self.evaluation_results.append({
                        "query": query,
                        "category": category,
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Calculate overall metrics
        total_time = time.time() - evaluation_start
        successful_queries = sum(1 for r in self.evaluation_results if r.get("success"))
        
        metrics = {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": (successful_queries / total_queries) * 100,
            "total_evaluation_time": total_time,
            "average_response_time": sum(r.get("response_time", 0) for r in self.evaluation_results if "response_time" in r) / len([r for r in self.evaluation_results if "response_time" in r]),
            "average_quality_score": sum(r.get("quality_score", 0) for r in self.evaluation_results if "quality_score" in r) / len([r for r in self.evaluation_results if "quality_score" in r]),
            "category_breakdown": self._calculate_category_metrics(),
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n{'='*60}")
        print(f"🎯 EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total Queries: {total_queries}")
        print(f"Successful: {successful_queries} ({metrics['success_rate']:.1f}%)")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Avg Response Time: {metrics['average_response_time']:.2f}s")
        print(f"Avg Quality Score: {metrics['average_quality_score']:.1f}/10")
        
        return {
            "metrics": metrics,
            "detailed_results": self.evaluation_results
        }
    
    def _evaluate_response_quality(self, query: str, result: Dict) -> float:
        """
        Evaluate response quality on a scale of 1-10
        
        Args:
            query: Original user query
            result: RAG response result
            
        Returns:
            Quality score from 1-10
        """
        if not result.get("success"):
            return 1.0
        
        answer = result.get("answer", "").lower()
        query_lower = query.lower()
        
        score = 5.0  # Base score
        
        # Check for appropriate response length
        answer_length = len(result.get("answer", ""))
        if 50 <= answer_length <= 500:
            score += 1.0
        elif answer_length > 500:
            score += 0.5
        
        # Check for report citations
        if "report" in answer and any(char.isdigit() for char in answer):
            score += 1.0
        
        # Check for appropriate guardrails
        if any(phrase in answer for phrase in ["i can only", "based on the available reports", "i'm sorry, i can only"]):
            score += 1.0
        
        # Check for refusal of inappropriate queries
        if any(phrase in query_lower for phrase in ["weather", "cook", "2+2", "mars"]):
            if any(phrase in answer for phrase in ["i can only", "detection reports", "i'm sorry"]):
                score += 2.0
            else:
                score -= 2.0
        
        # Check for specific information extraction
        if any(phrase in query_lower for phrase in ["how many", "count", "total"]):
            if any(char.isdigit() for char in answer):
                score += 1.0
        
        # Check for time-based queries
        if any(phrase in query_lower for phrase in ["yesterday", "today", "last week", "last night"]):
            if any(phrase in answer for phrase in ["yesterday", "today", "week", "night", "recent"]):
                score += 1.0
        
        # Ensure score is within bounds
        return max(1.0, min(10.0, score))
    
    def _calculate_category_metrics(self) -> Dict[str, Dict]:
        """Calculate metrics for each query category"""
        category_metrics = {}
        
        for category in self.test_queries.keys():
            category_results = [r for r in self.evaluation_results if r.get("category") == category]
            
            if category_results:
                successful = sum(1 for r in category_results if r.get("success"))
                avg_time = sum(r.get("response_time", 0) for r in category_results if "response_time" in r) / len([r for r in category_results if "response_time" in r])
                avg_quality = sum(r.get("quality_score", 0) for r in category_results if "quality_score" in r) / len([r for r in category_results if "quality_score" in r])
                
                category_metrics[category] = {
                    "total_queries": len(category_results),
                    "success_rate": (successful / len(category_results)) * 100,
                    "average_response_time": avg_time,
                    "average_quality_score": avg_quality
                }
        
        return category_metrics
    
    def save_results(self, filename: str = None):
        """Save evaluation results to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_evaluation_{timestamp}.json"
        
        results = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(self.evaluation_results),
                "evaluator_version": "1.0.0"
            },
            "detailed_results": self.evaluation_results
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")
        return filename

async def main():
    """Main evaluation function"""
    print("🚀 RAG Evaluation with PromptLayer Integration")
    print("=" * 60)
    
    # Check if PromptLayer is available
    try:
        import promptlayer
        print("✅ PromptLayer integration detected")
    except ImportError:
        print("⚠️  PromptLayer not installed - install with: pip install promptlayer")
        return
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Run evaluation
    results = await evaluator.run_evaluation()
    
    # Save results
    filename = evaluator.save_results()
    
    # Print summary
    print(f"\n📊 Evaluation Summary:")
    print(f"   Success Rate: {results['metrics']['success_rate']:.1f}%")
    print(f"   Avg Response Time: {results['metrics']['average_response_time']:.2f}s")
    print(f"   Avg Quality Score: {results['metrics']['average_quality_score']:.1f}/10")
    
    print(f"\n🔗 View detailed metrics in PromptLayer dashboard:")
    print(f"   https://promptlayer.com/dashboard")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
