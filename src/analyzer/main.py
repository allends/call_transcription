import os
from collections import Counter
import re
from datetime import datetime
import csv

def clean_text(text):
    # Remove special characters but keep apostrophes
    text = re.sub(r'[^\w\s\']', ' ', text.lower())
    return text

def analyze_transcripts(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    text_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    print(f"=== Analyzing {len(text_files)} transcripts ===\n")
    
    topics = Counter()
    business_phrases = Counter()
    
    # More specific business-focused keywords
    topic_keywords = {
        'Order Issues': ['order status', 'where is my order', 'missing item', 'wrong item', 'damaged', 'delivery issue', 'shipping delay'],
        'Payment Problems': ['refund', 'charge', 'payment declined', 'double charged', 'pricing', 'discount', 'coupon'],
        'Lead Questions': ['quote', 'proposal', 'interested in', 'looking to buy', 'pricing for', 'cost of'],
        'Product Inquiries': ['specifications', 'dimensions', 'compatibility', 'difference between', 'compare'],
        'Website Technical': ['cant checkout', 'error message', 'website down', 'payment failed', 'login issues'],
        'Order Modifications': ['cancel order', 'change address', 'modify order', 'update shipping'],
        'Returns/Refunds': ['return policy', 'exchange', 'return label', 'money back'],
        'Stock Issues': ['out of stock', 'backorder', 'availability', 'when will'],
        'Account Management': ['reset password', 'update account', 'login problem', 'account locked'],
        'Shipping Questions': ['shipping cost', 'delivery time', 'tracking number', 'shipping method']
    }

    # Business-specific phrases to track
    business_phrase_patterns = [
        r'need\s+(?:to|a)\s+(?:refund|return|exchange)',
        r'(?:having|have)\s+(?:trouble|problem|issue)\s+with',
        r'(?:can|could)\s+(?:you|someone)\s+help\s+(?:me|with)',
        r'how\s+(?:long|much)\s+(?:does|will|is)',
        r'want\s+to\s+(?:order|buy|purchase|return)',
        r'(?:tracking|order)\s+number\s+is',
        r'(?:didn\'t|did\s+not)\s+receive',
        r'when\s+will\s+(?:my|the)\s+(?:order|item|product)',
        r'(?:looking|interested)\s+in\s+(?:buying|purchasing)',
        r'need\s+(?:help|assistance)\s+with'
    ]

    print("Processing transcripts...")
    for file_name in text_files:
        with open(os.path.join(input_folder, file_name), 'r') as file:
            try:
                text = file.read().lower()
                clean_content = clean_text(text)
                
                # Check for topics
                for topic, keywords in topic_keywords.items():
                    if any(keyword.lower() in clean_content for keyword in keywords):
                        topics[topic] += 1
                
                # Find business-specific phrases
                for pattern in business_phrase_patterns:
                    matches = re.findall(pattern, clean_content)
                    for match in matches:
                        if len(match) > 5:  # Avoid very short matches
                            business_phrases[match.strip()] += 1
                
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
    
    # Generate CSV report with today's date
    today = datetime.now().strftime('%Y-%m-%d')
    output_file = os.path.join(output_folder, f'analysis_report_{today}.csv')
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header and summary
        writer.writerow(['Analysis Report', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow(['Total Calls Analyzed', len(text_files)])
        writer.writerow([])
        
        # Write topic breakdown
        writer.writerow(['BUSINESS TOPIC BREAKDOWN'])
        writer.writerow(['Topic', 'Count', 'Percentage'])
        if topics:
            for topic, count in topics.most_common():
                percentage = (count / len(text_files)) * 100
                writer.writerow([topic, count, f'{percentage:.1f}%'])
        else:
            writer.writerow(['No topics found in analyzed transcripts', '-', '-'])
        
        writer.writerow([])
        
        # Write key business phrases
        writer.writerow(['KEY BUSINESS PHRASES'])
        writer.writerow(['Phrase', 'Count'])
        has_phrases = False
        for phrase, count in business_phrases.most_common(15):
            if count > 2:
                writer.writerow([phrase, count])
                has_phrases = True
        if not has_phrases:
            writer.writerow(['No significant business phrases found', '-'])
    
    print(f"\nAnalysis complete. Report saved to: {output_file}")

if __name__ == "__main__":
    input_folder = "processed_transcriptions"
    output_folder = "analysis"
    analyze_transcripts(input_folder, output_folder)