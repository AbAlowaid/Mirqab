"""
Moraqib RAG System - Intelligent Retrieval-Augmented Generation
Handles complex queries about detection reports with smart data retrieval
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import re

# Load environment variables
load_dotenv()


class MoraqibRAG:
    """
    Advanced RAG system for querying detection reports
    Features:
    - Intent classification (summary, aggregation, filtering, general query)
    - Smart temporal parsing (last report, yesterday, last week, etc.)
    - Aggregation capabilities (average, count, list)
    - Context-aware response generation
    """
    
    def __init__(self, firestore_handler):
        """
        Initialize Moraqib RAG system
        
        Args:
            firestore_handler: Instance of FirestoreHandler for database access
        """
        self.firestore = firestore_handler
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            print("⚠️ Warning: OPENAI_API_KEY not found in environment")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
            print("✅ Moraqib RAG initialized with OpenAI")
    
    async def query(self, user_query: str) -> Dict:
        """
        Main query handler - orchestrates the RAG pipeline
        
        Args:
            user_query: Natural language question from user
        
        Returns:
            Dict with answer, metadata, and context
        """
        print(f"\n{'='*60}")
        print(f"🔍 Processing query: {user_query}")
        print(f"{'='*60}")
        
        # Step 1: Understand the query intent
        intent = self._classify_intent(user_query)
        print(f"📊 Query intent: {intent['type']}")
        
        # Step 2: Extract temporal information
        temporal_context = self._extract_temporal_context(user_query)
        print(f"📅 Temporal context: {temporal_context}")
        
        # Step 3: Retrieve relevant reports
        reports = self._retrieve_reports(intent, temporal_context, user_query)
        print(f"📚 Retrieved {len(reports)} reports")
        
        # Step 4: Generate response using LLM with retrieved context
        answer = await self._generate_answer(user_query, reports, intent, temporal_context)
        
        # Extract report IDs used
        report_ids = [r.get('report_id', 'Unknown') for r in reports]
        
        return {
            "success": True,
            "question": user_query,
            "answer": answer,
            "reports_count": len(reports),
            "reports_used": report_ids[:10],  # Limit to first 10 for display
            "intent": intent['type']
        }
    
    def _classify_intent(self, query: str) -> Dict:
        """
        Classify the user's query intent
        
        Types:
        - summary: "Give me summary of...", "Describe...", "Tell me about..."
        - aggregation: "How many...", "Average...", "Total...", "Count..."
        - filtering: "Show reports from...", "List devices...", "Find..."
        - latest: "Last report", "Most recent", "Latest..."
        - general: General questions about data
        
        Args:
            query: User's question
        
        Returns:
            Dict with intent type and confidence
        """
        query_lower = query.lower()
        
        # Aggregation patterns
        aggregation_keywords = [
            'how many', 'count', 'total', 'average', 'avg', 'sum',
            'number of', 'calculate', 'statistics', 'stat'
        ]
        
        # Summary patterns
        summary_keywords = [
            'summary', 'summarize', 'describe', 'tell me about',
            'what is', 'what was', 'explain', 'overview'
        ]
        
        # Latest/Last patterns
        latest_keywords = [
            'last report', 'latest', 'most recent', 'newest',
            'last detection', 'recent report'
        ]
        
        # Filtering patterns
        filtering_keywords = [
            'show', 'list', 'find', 'search', 'get',
            'from device', 'by device', 'where'
        ]
        
        # Device-specific queries
        device_keywords = [
            'device', 'devices', 'pi-', 'raspberry', 'sensor'
        ]
        
        # Check for aggregation
        if any(kw in query_lower for kw in aggregation_keywords):
            return {
                'type': 'aggregation',
                'confidence': 0.9
            }
        
        # Check for latest/last
        if any(kw in query_lower for kw in latest_keywords):
            return {
                'type': 'latest',
                'confidence': 0.95
            }
        
        # Check for summary
        if any(kw in query_lower for kw in summary_keywords):
            return {
                'type': 'summary',
                'confidence': 0.85
            }
        
        # Check for device queries
        if any(kw in query_lower for kw in device_keywords):
            return {
                'type': 'device_query',
                'confidence': 0.9
            }
        
        # Check for filtering
        if any(kw in query_lower for kw in filtering_keywords):
            return {
                'type': 'filtering',
                'confidence': 0.8
            }
        
        # Default to general query
        return {
            'type': 'general',
            'confidence': 0.5
        }
    
    def _extract_temporal_context(self, query: str) -> Dict:
        """
        Extract time-related information from query
        
        Patterns:
        - "last report" -> most recent 1
        - "yesterday" -> reports from yesterday
        - "last week" -> reports from last 7 days
        - "last 10 reports" -> most recent 10
        - "today" -> reports from today
        - "13th report", "3rd report", "1st report" -> specific ordinal position
        
        Args:
            query: User's question
        
        Returns:
            Dict with temporal filters
        """
        query_lower = query.lower()
        now = datetime.now()
        
        # Check for ordinal numbers (1st, 2nd, 3rd, 13th, etc.)
        # Matches: "1st report", "2nd report", "3rd report", "13th report", "13rd report" (typo)
        ordinal_match = re.search(r'(\d+)(?:st|nd|rd|th) report', query_lower)
        if ordinal_match:
            n = int(ordinal_match.group(1))
            return {
                'type': 'ordinal',
                'ordinal_position': n,
                'limit': n,  # Get up to Nth report
                'start_date': None,
                'end_date': None
            }
        
        # Check for "last N reports"
        last_n_match = re.search(r'last (\d+) reports?', query_lower)
        if last_n_match:
            n = int(last_n_match.group(1))
            return {
                'type': 'last_n',
                'limit': n,
                'start_date': None,
                'end_date': None
            }
        
        # Check for "last report" (singular)
        if 'last report' in query_lower or 'latest report' in query_lower or 'most recent report' in query_lower:
            return {
                'type': 'last_n',
                'limit': 1,
                'start_date': None,
                'end_date': None
            }
        
        # Check for time ranges
        if 'today' in query_lower:
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return {
                'type': 'date_range',
                'limit': 1000,
                'start_date': start_of_day,
                'end_date': now
            }
        
        if 'yesterday' in query_lower:
            yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday_end = yesterday_start + timedelta(days=1)
            return {
                'type': 'date_range',
                'limit': 1000,
                'start_date': yesterday_start,
                'end_date': yesterday_end
            }
        
        if 'last week' in query_lower or 'past week' in query_lower:
            start_date = now - timedelta(days=7)
            return {
                'type': 'date_range',
                'limit': 1000,
                'start_date': start_date,
                'end_date': now
            }
        
        if 'last month' in query_lower or 'past month' in query_lower:
            start_date = now - timedelta(days=30)
            return {
                'type': 'date_range',
                'limit': 1000,
                'start_date': start_date,
                'end_date': now
            }
        
        # Check for "last N days"
        days_match = re.search(r'last (\d+) days?', query_lower)
        if days_match:
            n_days = int(days_match.group(1))
            start_date = now - timedelta(days=n_days)
            return {
                'type': 'date_range',
                'limit': 1000,
                'start_date': start_date,
                'end_date': now
            }
        
        # Default: return recent reports
        return {
            'type': 'recent',
            'limit': 100,
            'start_date': None,
            'end_date': None
        }
    
    def _retrieve_reports(
        self,
        intent: Dict,
        temporal_context: Dict,
        query: str
    ) -> List[Dict]:
        """
        Retrieve relevant reports from Firestore based on intent and context
        
        Args:
            intent: Query intent classification
            temporal_context: Temporal filters
            query: Original user query
        
        Returns:
            List of relevant report dictionaries
        """
        # Extract device filter if mentioned
        device_id = self._extract_device_id(query)
        
        # Determine query parameters
        limit = temporal_context.get('limit', 100)
        start_date = temporal_context.get('start_date')
        end_date = temporal_context.get('end_date')
        
        # For ordinal queries, sort ascending (oldest first) so 1st = oldest
        # For other queries, sort descending (newest first)
        is_ordinal = temporal_context.get('type') == 'ordinal'
        order_by_desc = not is_ordinal  # Reverse for ordinal queries
        
        print(f"🔍 Querying Firestore:")
        print(f"   Device: {device_id or 'All'}")
        print(f"   Limit: {limit}")
        print(f"   Order: {'Ascending (oldest first)' if not order_by_desc else 'Descending (newest first)'}")
        print(f"   Date range: {start_date} to {end_date}")
        
        # Query Firestore
        try:
            if not self.firestore.db:
                print("❌ Firestore not initialized")
                return []
            
            reports = self.firestore.query_reports(
                start_date=start_date,
                end_date=end_date,
                device_id=device_id,
                limit=limit,
                order_by_desc=order_by_desc
            )
            
            return reports
        
        except Exception as e:
            print(f"❌ Error retrieving reports: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _extract_device_id(self, query: str) -> Optional[str]:
        """
        Extract device ID from query if mentioned
        
        Args:
            query: User's question
        
        Returns:
            Device ID string or None
        """
        query_lower = query.lower()
        
        # Look for patterns like "Pi-001", "device Pi-001", "from Pi-001"
        device_match = re.search(r'pi-\d{3}', query_lower, re.IGNORECASE)
        if device_match:
            return device_match.group(0).upper()
        
        # Look for "web upload" or "web-upload"
        if 'web' in query_lower and ('upload' in query_lower or 'uploaded' in query_lower):
            return "Web-Upload"
        
        return None
    
    async def _generate_answer(
        self,
        query: str,
        reports: List[Dict],
        intent: Dict,
        temporal_context: Dict
    ) -> str:
        """
        Generate natural language answer using OpenAI with retrieved context
        
        Args:
            query: Original user question
            reports: Retrieved reports from database
            intent: Query intent classification
            temporal_context: Temporal context
        
        Returns:
            Natural language answer
        """
        if not self.client:
            return self._generate_fallback_answer(query, reports, intent, temporal_context)
        
        # Handle empty results
        if not reports:
            return self._handle_no_results(query, temporal_context)
        
        # Special handling for ordinal queries (1st, 2nd, 13th report, etc.)
        if temporal_context.get('type') == 'ordinal':
            ordinal_pos = temporal_context.get('ordinal_position', 1)
            if ordinal_pos <= len(reports):
                # Get the specific report at ordinal position
                target_report = reports[ordinal_pos - 1]  # Convert to 0-based index
                
                # Create focused context for this specific report
                context = self._prepare_ordinal_context(target_report, ordinal_pos)
            else:
                return f"❌ Only {len(reports)} reports exist in the database. Cannot retrieve report #{ordinal_pos}."
        else:
            # Prepare context from reports
            context = self._prepare_context(reports, intent)
        
        # Create system prompt
        system_prompt = """You are Moraqib, an AI assistant specialized in analyzing military detection reports from the Mirqab system.

Your role:
- Answer questions ONLY based on the detection reports provided in the context
- Be precise with numbers and statistics
- Always cite report IDs when referencing specific reports
- Format responses clearly with bullet points or structured text when appropriate
- If the data doesn't contain the answer, say so clearly

Detection Report Schema:
- report_id: Unique identifier (e.g., MIR-20251027-0001)
- timestamp: When the detection occurred
- location: GPS coordinates (latitude, longitude)
- soldier_count: Number of camouflaged soldiers detected
- environment: Description of the environment (e.g., "dense woodland", "urban area")
- attire_and_camouflage: Description of camouflage patterns
- equipment: Visible equipment
- source_device_id: Device that made the detection (e.g., "Pi-001", "Web-Upload")

Guidelines:
- Be concise but informative
- Use markdown formatting for better readability
- Present statistics clearly
- Group related information together"""

        # Create user prompt with context
        if temporal_context and temporal_context.get('type') == 'ordinal':
            ordinal_pos = temporal_context.get('ordinal_position', 1)
            ordinal_suffix = 'th'
            if ordinal_pos % 10 == 1 and ordinal_pos % 100 != 11:
                ordinal_suffix = 'st'
            elif ordinal_pos % 10 == 2 and ordinal_pos % 100 != 12:
                ordinal_suffix = 'nd'
            elif ordinal_pos % 10 == 3 and ordinal_pos % 100 != 13:
                ordinal_suffix = 'rd'
            
            user_prompt = f"""Question: {query}

IMPORTANT: The user is asking about the {ordinal_pos}{ordinal_suffix} report in chronological order (sorted by timestamp, oldest first).
This means: 1st report = the very first/oldest report, 2nd = second oldest, etc.
Report position #{ordinal_pos} counting from the oldest.

{context}

Please provide a detailed summary of this specific report. Include all relevant details: report ID, timestamp, soldiers, environment, attire, equipment, device, and location."""
        else:
            user_prompt = f"""Question: {query}

Context - Detection Reports ({len(reports)} reports, sorted newest first):
{context}

Please answer the question based on the detection reports above. Be specific and cite report IDs when relevant."""

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            print(f"✅ Generated answer ({len(answer)} chars)")
            
            return answer
        
        except Exception as e:
            print(f"❌ Error generating answer with OpenAI: {e}")
            return self._generate_fallback_answer(query, reports, intent, temporal_context)
    
    def _prepare_ordinal_context(self, report: Dict, position: int) -> str:
        """
        Prepare context for a specific ordinal report (1st, 2nd, 13th, etc.)
        
        Args:
            report: Single report dictionary
            position: Ordinal position (1st, 2nd, 13th, etc.)
        
        Returns:
            Formatted context string focused on this specific report
        """
        location = report.get('location', {})
        
        ordinal_suffix = 'th'
        if position % 10 == 1 and position % 100 != 11:
            ordinal_suffix = 'st'
        elif position % 10 == 2 and position % 100 != 12:
            ordinal_suffix = 'nd'
        elif position % 10 == 3 and position % 100 != 13:
            ordinal_suffix = 'rd'
        
        context = f"""This is the {position}{ordinal_suffix} report in the database (sorted by timestamp, oldest first):

Report ID: {report.get('report_id', 'Unknown')}
Position: #{position} (1st = oldest report, chronologically)
Timestamp: {report.get('timestamp', 'Unknown')}
Soldier Count: {report.get('soldier_count', 0)}
Environment: {report.get('environment', 'Unknown')}
Attire and Camouflage: {report.get('attire_and_camouflage', 'Unknown')}
Equipment: {report.get('equipment', 'Unknown')}
Source Device: {report.get('source_device_id', 'Unknown')}
Location: ({location.get('latitude', 0)}, {location.get('longitude', 0)})
"""
        return context
    
    def _prepare_context(self, reports: List[Dict], intent: Dict) -> str:
        """
        Prepare context string from reports for LLM
        
        Args:
            reports: List of report dictionaries
            intent: Query intent
        
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # For aggregation queries, provide summary statistics first
        if intent['type'] == 'aggregation':
            stats = self._calculate_statistics(reports)
            context_parts.append(f"Summary Statistics:")
            context_parts.append(f"- Total Reports: {stats['total_reports']}")
            context_parts.append(f"- Total Soldiers Detected: {stats['total_soldiers']}")
            context_parts.append(f"- Average Soldiers per Report: {stats['avg_soldiers']:.2f}")
            context_parts.append(f"- Unique Devices: {stats['unique_devices']}")
            context_parts.append(f"- Devices List: {', '.join(stats['device_list'])}")
            context_parts.append("\n")
        
        # Add individual reports (limit to 50 for context window)
        context_parts.append("Individual Reports:")
        for i, report in enumerate(reports[:50], 1):
            location = report.get('location', {})
            context_parts.append(f"\n{i}. Report: {report.get('report_id', 'Unknown')}")
            context_parts.append(f"   Time: {report.get('timestamp', 'Unknown')}")
            context_parts.append(f"   Soldiers: {report.get('soldier_count', 0)}")
            context_parts.append(f"   Environment: {report.get('environment', 'Unknown')}")
            context_parts.append(f"   Attire: {report.get('attire_and_camouflage', 'Unknown')}")
            context_parts.append(f"   Equipment: {report.get('equipment', 'Unknown')}")
            context_parts.append(f"   Device: {report.get('source_device_id', 'Unknown')}")
            context_parts.append(f"   Location: ({location.get('latitude', 0)}, {location.get('longitude', 0)})")
        
        if len(reports) > 50:
            context_parts.append(f"\n... and {len(reports) - 50} more reports")
        
        return "\n".join(context_parts)
    
    def _calculate_statistics(self, reports: List[Dict]) -> Dict:
        """
        Calculate statistics from reports
        
        Args:
            reports: List of report dictionaries
        
        Returns:
            Dictionary of statistics
        """
        total_soldiers = sum(r.get('soldier_count', 0) for r in reports)
        devices = set(r.get('source_device_id', 'Unknown') for r in reports)
        
        return {
            'total_reports': len(reports),
            'total_soldiers': total_soldiers,
            'avg_soldiers': total_soldiers / len(reports) if reports else 0,
            'unique_devices': len(devices),
            'device_list': sorted(list(devices))
        }
    
    def _generate_fallback_answer(
        self,
        query: str,
        reports: List[Dict],
        intent: Dict,
        temporal_context: Dict = None
    ) -> str:
        """
        Generate answer without LLM (fallback mode)
        
        Args:
            query: User's question
            reports: Retrieved reports
            intent: Query intent
            temporal_context: Temporal context (optional)
        
        Returns:
            Basic formatted answer
        """
        if not reports:
            return "No reports found matching your query."
        
        # Handle ordinal queries
        if temporal_context and temporal_context.get('type') == 'ordinal':
            ordinal_pos = temporal_context.get('ordinal_position', 1)
            if ordinal_pos <= len(reports):
                report = reports[ordinal_pos - 1]
                location = report.get('location', {})
                
                ordinal_suffix = 'th'
                if ordinal_pos % 10 == 1 and ordinal_pos % 100 != 11:
                    ordinal_suffix = 'st'
                elif ordinal_pos % 10 == 2 and ordinal_pos % 100 != 12:
                    ordinal_suffix = 'nd'
                elif ordinal_pos % 10 == 3 and ordinal_pos % 100 != 13:
                    ordinal_suffix = 'rd'
                
                return f"""Here is the {ordinal_pos}{ordinal_suffix} report (chronologically, oldest first):

Report: {report.get('report_id', 'Unknown')}
Time: {report.get('timestamp', 'Unknown')}
Soldiers: {report.get('soldier_count', 0)}
Environment: {report.get('environment', 'Unknown')}
Attire: {report.get('attire_and_camouflage', 'Unknown')}
Equipment: {report.get('equipment', 'Unknown')}
Device: {report.get('source_device_id', 'Unknown')}
Location: ({location.get('latitude', 0)}, {location.get('longitude', 0)})

(LLM unavailable - showing basic report details)"""
            else:
                return f"❌ Only {len(reports)} reports exist. Cannot retrieve report #{ordinal_pos}."
        
        # Calculate basic statistics
        stats = self._calculate_statistics(reports)
        
        # Generate response based on intent
        if intent['type'] == 'aggregation':
            return f"""Based on {stats['total_reports']} reports:
- Total soldiers detected: {stats['total_soldiers']}
- Average soldiers per report: {stats['avg_soldiers']:.2f}
- Devices used: {', '.join(stats['device_list'])}

(LLM unavailable - showing basic statistics)"""
        
        elif intent['type'] == 'latest':
            report = reports[0]
            location = report.get('location', {})
            return f"""Latest Report: {report.get('report_id', 'Unknown')}
- Time: {report.get('timestamp', 'Unknown')}
- Soldiers detected: {report.get('soldier_count', 0)}
- Environment: {report.get('environment', 'Unknown')}
- Attire: {report.get('attire_and_camouflage', 'Unknown')}
- Device: {report.get('source_device_id', 'Unknown')}
- Location: ({location.get('latitude', 0)}, {location.get('longitude', 0)})"""
        
        else:
            # General response
            report_ids = [r.get('report_id', 'Unknown') for r in reports[:5]]
            return f"""Found {len(reports)} reports matching your query.

Recent reports: {', '.join(report_ids)}

(LLM unavailable - showing report list only)"""
    
    def _handle_no_results(self, query: str, temporal_context: Dict) -> str:
        """
        Handle case when no reports are found
        
        Args:
            query: User's question
            temporal_context: Temporal filters used
        
        Returns:
            Helpful message
        """
        time_desc = ""
        if temporal_context['type'] == 'last_n':
            time_desc = f"the last {temporal_context['limit']} reports"
        elif temporal_context['type'] == 'date_range':
            if temporal_context['start_date']:
                time_desc = f"between {temporal_context['start_date'].strftime('%Y-%m-%d')} and {temporal_context['end_date'].strftime('%Y-%m-%d')}"
        
        return f"""No detection reports found {time_desc}.

This could mean:
- No detections have been recorded yet
- The time range specified has no data
- The device filter doesn't match any reports

Try:
- Asking about "all reports" or "all detections"
- Expanding your time range
- Checking if the database has been populated with detection data"""


# Global instance
moraqib_rag = None


def initialize_rag(firestore_handler):
    """Initialize the global RAG instance"""
    global moraqib_rag
    moraqib_rag = MoraqibRAG(firestore_handler)
    return moraqib_rag

