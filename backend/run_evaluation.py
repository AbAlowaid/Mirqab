#!/usr/bin/env python3
"""
Quick RAG Evaluation Runner
Sets up environment and runs evaluation with PromptLayer integration
"""

import os
import sys
import asyncio
from pathlib import Path

def setup_environment():
    """Set up environment variables for evaluation"""
    print("🔧 Setting up environment...")
    
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check if PROMPTLAYER_API_KEY is set
    if not os.getenv("PROMPTLAYER_API_KEY"):
        print("⚠️ Warning: PROMPTLAYER_API_KEY not found in environment")
        print("   Please set it in your .env file or environment variables")
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    print("✅ Environment configured")

async def run_evaluation():
    """Run the RAG evaluation"""
    try:
        from rag_evaluation import RAGEvaluator
        
        print("🚀 Starting RAG Evaluation...")
        evaluator = RAGEvaluator()
        results = await evaluator.run_evaluation()
        
        # Save results
        filename = evaluator.save_results()
        
        print(f"\n🎯 Evaluation Complete!")
        print(f"📊 Results saved to: {filename}")
        print(f"🔗 View PromptLayer metrics: https://promptlayer.com/dashboard")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    print("=" * 60)
    print("🔬 RAG Evaluation with PromptLayer Integration")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Run evaluation
    results = asyncio.run(run_evaluation())
    
    if results:
        print("\n✅ Evaluation completed successfully!")
        print("📊 Check the PromptLayer dashboard for detailed metrics")
    else:
        print("\n❌ Evaluation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
