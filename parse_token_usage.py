#!/usr/bin/env python3
"""
Token Usage Log Parser

This script parses log files containing token usage information from llama-stack
and provides detailed statistics about token consumption.

Usage:
    python parse_token_usage.py [logfile]
    
If no logfile is provided, it reads from stdin.

Example log format it parses:
    WARNING AGENT_TOKEN_USAGE|gemini/gemini-2.5-flash|session-123|{'prompt_tokens': 204, 'completion_tokens': 27, 'total_tokens': 231}
"""

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import ast


class TokenUsageStats:
    """Class to track and analyze token usage statistics."""
    
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0
        self.streaming_count = 0
        self.non_streaming_count = 0
        
        # Breakdown by model
        self.model_stats = defaultdict(lambda: {
            'prompt_tokens': 0, 
            'completion_tokens': 0, 
            'total_tokens': 0, 
            'requests': 0,
            'streaming_requests': 0
        })
        
        # Breakdown by conversation
        self.conversation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'prompt_tokens': 0, 
            'completion_tokens': 0, 
            'total_tokens': 0, 
            'requests': 0,
            'models_used': set()
        })
        
        # Track individual requests for detailed analysis
        self.requests = []
        
    def add_request(self, model: str, conversation: str, metrics: Dict[str, int], is_streaming: bool, timestamp: Optional[str] = None):
        """Add a token usage request to the statistics."""
        prompt_tokens = metrics.get('prompt_tokens', 0)
        completion_tokens = metrics.get('completion_tokens', 0) 
        total_tokens = metrics.get('total_tokens', 0)
        
        # Update totals
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.request_count += 1
        
        if is_streaming:
            self.streaming_count += 1
        else:
            self.non_streaming_count += 1
            
        # Update model stats
        model_stat = self.model_stats[model]
        model_stat['prompt_tokens'] += prompt_tokens
        model_stat['completion_tokens'] += completion_tokens
        model_stat['total_tokens'] += total_tokens
        model_stat['requests'] += 1
        if is_streaming:
            model_stat['streaming_requests'] += 1
            
        # Update conversation stats
        conv_stat = self.conversation_stats[conversation]
        conv_stat['prompt_tokens'] += prompt_tokens
        conv_stat['completion_tokens'] += completion_tokens
        conv_stat['total_tokens'] += total_tokens
        conv_stat['requests'] += 1
        conv_stat['models_used'].add(model)
        
        # Store individual request
        self.requests.append({
            'timestamp': timestamp,
            'model': model,
            'conversation': conversation,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'is_streaming': is_streaming
        })


def parse_log_line(line: str) -> Optional[Tuple[str, str, Dict[str, int], bool, str]]:
    """
    Parse a single log line and extract token usage information.
    
    Handles llama-stack format: AGENT_TOKEN_USAGE|model|session_id|metrics_dict
    
    Returns:
        (model, conversation_id, metrics_dict, is_streaming, timestamp)
    """
    # Parse pipe-separated format from llama-stack agent
    agent_pattern = r'.*?AGENT_TOKEN_USAGE\|([^|]+)\|([^|]+)\|(.+)'
    match = re.search(agent_pattern, line)
    if not match:
        return None
        
    model, session_id, metrics_str = match.groups()
    
    # Extract timestamp if present
    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})', line)
    timestamp = timestamp_match.group(1) if timestamp_match else "Unknown"
    
    try:
        # Parse the metrics dictionary
        metrics = ast.literal_eval(metrics_str.strip())
        if isinstance(metrics, dict):
            return model.strip(), session_id.strip(), metrics, False, timestamp
    except (ValueError, SyntaxError):
        pass
        
    return None


def parse_log_file(file_handle) -> TokenUsageStats:
    """Parse a log file and return token usage statistics."""
    stats = TokenUsageStats()
    line_count = 0
    parsed_count = 0
    
    # Read all lines to handle multi-line log entries
    lines = [line.rstrip() for line in file_handle]
    reconstructed_lines = reconstruct_multiline_logs(lines)
    
    for line in reconstructed_lines:
        line_count += 1
        result = parse_log_line(line.strip())
        if result:
            model, conversation, metrics, is_streaming, timestamp = result
            stats.add_request(model, conversation, metrics, is_streaming, timestamp)
            parsed_count += 1
            
    print(f"Processed {line_count} log lines, found {parsed_count} token usage entries\n", file=sys.stderr)
    return stats


def reconstruct_multiline_logs(lines: List[str]) -> List[str]:
    """
    Reconstruct log entries that have been wrapped across multiple lines.
    
    Specifically handles cases where AGENT_TOKEN_USAGE data is split across lines.
    """
    reconstructed = []
    current_entry = ""
    in_token_usage_entry = False
    
    for line in lines:
        # Check if this line contains AGENT_TOKEN_USAGE
        if 'AGENT_TOKEN_USAGE' in line:
            # If we were building an entry, save it first
            if current_entry:
                reconstructed.append(current_entry)
            
            current_entry = line
            in_token_usage_entry = True
            
            # Check if this line is complete (contains closing brace)
            if line.count('{') == line.count('}') and '{' in line:
                reconstructed.append(current_entry)
                current_entry = ""
                in_token_usage_entry = False
                
        elif in_token_usage_entry:
            # This is a continuation of a token usage entry
            current_entry += " " + line.strip()
            
            # Check if entry is now complete
            if current_entry.count('{') == current_entry.count('}') and '{' in current_entry:
                reconstructed.append(current_entry)
                current_entry = ""
                in_token_usage_entry = False
                
        else:
            # Regular log line, not part of token usage
            if current_entry:
                reconstructed.append(current_entry)
                current_entry = ""
                in_token_usage_entry = False
            reconstructed.append(line)
    
    # Don't forget the last entry if we were building one
    if current_entry:
        reconstructed.append(current_entry)
    
    return reconstructed


def print_summary(stats: TokenUsageStats):
    """Print a comprehensive summary of token usage statistics."""
    
    print("=" * 80)
    print("TOKEN USAGE SUMMARY")
    print("=" * 80)
    
    # Overall totals
    print(f"Total Requests: {stats.request_count:,}")
    print(f"  - Non-streaming: {stats.non_streaming_count:,}")
    print(f"  - Streaming: {stats.streaming_count:,}")
    print()
    
    print(f"Total Token Usage:")
    print(f"  - Prompt tokens: {stats.total_prompt_tokens:,}")
    print(f"  - Completion tokens: {stats.total_completion_tokens:,}")
    print(f"  - Total tokens: {stats.total_tokens:,}")
    print()
    
    if stats.request_count > 0:
        avg_prompt = stats.total_prompt_tokens / stats.request_count
        avg_completion = stats.total_completion_tokens / stats.request_count
        avg_total = stats.total_tokens / stats.request_count
        
        print(f"Average per Request:")
        print(f"  - Prompt tokens: {avg_prompt:.1f}")
        print(f"  - Completion tokens: {avg_completion:.1f}")
        print(f"  - Total tokens: {avg_total:.1f}")
        print()
    
    # Model breakdown
    if stats.model_stats:
        print("=" * 80)
        print("BREAKDOWN BY MODEL")
        print("=" * 80)
        
        for model in sorted(stats.model_stats.keys()):
            model_stat = stats.model_stats[model]
            streaming_pct = (model_stat['streaming_requests'] / model_stat['requests'] * 100) if model_stat['requests'] > 0 else 0
            
            print(f"Model: {model}")
            print(f"  Requests: {model_stat['requests']:,} ({model_stat['streaming_requests']:,} streaming, {streaming_pct:.1f}%)")
            print(f"  Prompt tokens: {model_stat['prompt_tokens']:,}")
            print(f"  Completion tokens: {model_stat['completion_tokens']:,}")
            print(f"  Total tokens: {model_stat['total_tokens']:,}")
            
            if model_stat['requests'] > 0:
                avg_total = model_stat['total_tokens'] / model_stat['requests']
                print(f"  Average tokens per request: {avg_total:.1f}")
            print()
    
    # Conversation breakdown (top 10 by token usage)
    if stats.conversation_stats:
        print("=" * 80)
        print("TOP CONVERSATIONS BY TOKEN USAGE")
        print("=" * 80)
        
        # Sort conversations by total tokens used
        sorted_conversations = sorted(
            stats.conversation_stats.items(),
            key=lambda x: x[1]['total_tokens'],
            reverse=True
        )
        
        for i, (conv_id, conv_stat) in enumerate(sorted_conversations[:10]):
            models_used = ', '.join(sorted(conv_stat['models_used']))
            print(f"{i+1}. Conversation: {conv_id}")
            print(f"   Requests: {conv_stat['requests']:,}")
            print(f"   Total tokens: {conv_stat['total_tokens']:,}")
            print(f"   Models used: {models_used}")
            print()
    
    # Show token distribution
    if stats.requests:
        print("=" * 80)
        print("TOKEN USAGE DISTRIBUTION")
        print("=" * 80)
        
        token_counts = [req['total_tokens'] for req in stats.requests]
        token_counts.sort()
        
        percentiles = [50, 75, 90, 95, 99]
        print("Token usage percentiles:")
        for p in percentiles:
            idx = int((p / 100) * len(token_counts)) - 1
            if idx >= 0:
                print(f"  {p}th percentile: {token_counts[idx]:,} tokens")
        
        print(f"  Minimum: {min(token_counts):,} tokens")
        print(f"  Maximum: {max(token_counts):,} tokens")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Parse token usage logs and provide detailed statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'logfile', 
        nargs='?', 
        help='Log file to parse (reads from stdin if not provided)'
    )
    parser.add_argument(
        '--json', 
        action='store_true', 
        help='Output results in JSON format'
    )
    
    args = parser.parse_args()
    
    # Open input source
    if args.logfile:
        try:
            with open(args.logfile, 'r', encoding='utf-8') as f:
                stats = parse_log_file(f)
        except FileNotFoundError:
            print(f"Error: File '{args.logfile}' not found", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        stats = parse_log_file(sys.stdin)
    
    if args.json:
        import json
        output = {
            'total_requests': stats.request_count,
            'streaming_requests': stats.streaming_count,
            'non_streaming_requests': stats.non_streaming_count,
            'total_prompt_tokens': stats.total_prompt_tokens,
            'total_completion_tokens': stats.total_completion_tokens,
            'total_tokens': stats.total_tokens,
            'model_stats': dict(stats.model_stats),
            'conversation_count': len(stats.conversation_stats)
        }
        print(json.dumps(output, indent=2))
    else:
        print_summary(stats)


if __name__ == '__main__':
    main() 