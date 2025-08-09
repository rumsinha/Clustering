import pandas as pd
import json
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from umap import UMAP # pip3 install umap-learn
import matplotlib.pyplot as plt
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score, normalized_mutual_info_score,
                             homogeneity_score, completeness_score, v_measure_score)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
# Topic Modeling with BERTopic
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, davies_bouldin_score,
                             adjusted_rand_score, normalized_mutual_info_score,
                             homogeneity_score, completeness_score, v_measure_score)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
# Topic Modeling with BERTopic
from bertopic import BERTopic
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ========================= OLLAMA INTEGRATION =========================
# Using the requests library to communicate with the local Ollama server.
#
# Setting a low temperature and num_predict to ensure the model's output is factual and concise, not creative.
#
# Including a retry mechanism (max_retries) with a delay to handle potential API issues.
#
# Implementing a fallback mechanism to create a simple name from keywords if the LLM fails, ensuring the process doesn't break.
def generate_topic_name_with_ollama(keywords, sample_texts=None, model_name="llama3.2:3b", max_retries=3):
    """
    Generate meaningful topic names using Ollama and Llama model based on keywords and sample texts

    Args:
        keywords: List of (word, score) tuples from BERTopic
        sample_texts: Optional list of sample texts from the topic
        model_name: Ollama model name
        max_retries: Maximum number of retry attempts

    Returns:
        Generated topic name as string
    """
    # Extract just the words from keywords
    keyword_list = [word for word, score in keywords[:10]]  # Top 10 keywords
    keywords_str = ", ".join(keyword_list)

    # Build prompt with optional sample texts
    sample_text_part = ""
    if sample_texts:
        sample_text_part = f"\n\nSample texts from this topic:\n"
        for i, text in enumerate(sample_texts[:3]):  # Max 3 samples
            sample_text_part += f"{i+1}. {text[:150]}...\n"

    prompt = f"""You are an expert topic analyst. Your task is to create a concise, meaningful topic name based on the provided keywords and sample texts.

**Keywords:** {keywords_str}

{sample_text_part}

**Instructions:**
1. Analyze the keywords and sample texts to understand the main theme
2. Create a topic name that is:
   - Concise (2-4 words maximum)
   - Descriptive and specific
   - Professional and clear
   - Captures the essence of the topic
3. Return ONLY the topic name, nothing else
4. Do not include phrases like "Topic:", "The topic is:", etc.

**Examples:**
Keywords: technology, smartphone, apple, iphone, mobile → "Mobile Technology"
Keywords: politics, election, vote, candidate, campaign → "Election Politics" 
Keywords: health, covid, vaccine, virus, pandemic → "COVID Health"

Topic Name:"""

    for attempt in range(max_retries):
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.3,  # Slightly creative but focused
                        'num_predict': 15,   # Keep it short
                        'top_p': 0.9
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                topic_name = result['response'].strip()

                # Clean up the response
                topic_name = topic_name.split('\n')[0].strip()

                # Remove common prefixes if they appear
                prefixes_to_remove = ['Topic:', 'The topic is:', 'Topic Name:', 'Name:']
                for prefix in prefixes_to_remove:
                    if topic_name.startswith(prefix):
                        topic_name = topic_name[len(prefix):].strip()

                # Validate length and content
                if len(topic_name.split()) <= 6 and len(topic_name) > 0:
                    return topic_name
                else:
                    # Fallback: create name from top keywords
                    return " ".join(keyword_list[:2]).title()

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Fallback: create simple name from keywords
                return " ".join(keyword_list[:2]).title()
            time.sleep(1)  # Wait before retry

    # Final fallback
    return " ".join(keyword_list[:2]).title()

def batch_generate_topic_names(topic_model, news_df, model_name="llama3.2:3b",
                               exclude_outliers=True, use_sample_texts=True):
    """
    Generate topic names for all topics in batch

    Args:
        topic_model: Fitted BERTopic model
        news_df: DataFrame with topic assignments
        model_name: Ollama model name
        exclude_outliers: Whether to skip topic -1 (outliers)
        use_sample_texts: Whether to include sample texts in prompt

    Returns:
        Dictionary mapping topic_id to generated name
    """
    print("Generating topic names with Ollama...")

    topic_info = topic_model.get_topic_info()
    topic_names = {}

    # Filter topics
    topics_to_process = topic_info['Topic'].tolist()
    if exclude_outliers:
        topics_to_process = [t for t in topics_to_process if t != -1]

    print(f"Processing {len(topics_to_process)} topics...")

    for i, topic_id in enumerate(topics_to_process):
        print(f"Processing topic {topic_id} ({i+1}/{len(topics_to_process)})...")

        # Get keywords for this topic
        keywords = topic_model.get_topic(topic_id)

        if not keywords:
            topic_names[topic_id] = f"Topic {topic_id}"
            continue

        # Get sample texts if requested
        sample_texts = None
        if use_sample_texts:
            topic_docs = news_df[news_df['bertopic_topic'] == topic_id]
            if len(topic_docs) > 0:
                # Get diverse samples by taking from different clusters if available
                if 'cluster' in news_df.columns:
                    samples = []
                    for cluster in topic_docs['cluster'].unique()[:3]:
                        cluster_docs = topic_docs[topic_docs['cluster'] == cluster]
                        if len(cluster_docs) > 0:
                            samples.append(cluster_docs['text'].iloc[0])
                    sample_texts = samples
                else:
                    sample_texts = topic_docs['text'].head(3).tolist()

        # Generate topic name
        topic_name = generate_topic_name_with_ollama(
            keywords, sample_texts, model_name
        )

        topic_names[topic_id] = topic_name
        print(f"  Topic {topic_id}: {topic_name}")

        # Small delay to be nice to the API
        time.sleep(0.5)

    return topic_names

def update_topic_model_with_names(topic_model, topic_names):
    """
    Update BERTopic model with custom topic names

    Args:
        topic_model: BERTopic model
        topic_names: Dictionary mapping topic_id to name

    Returns:
        Updated topic model
    """
    print("Updating BERTopic model with custom names...")

    # Create custom labels
    custom_labels = {}
    for topic_id, name in topic_names.items():
        # Get keywords for context
        keywords = topic_model.get_topic(topic_id)
        if keywords:
            # Format: "Topic Name (top_keyword, keyword2, keyword3)"
            top_keywords = [word for word, score in keywords[:3]]
            custom_labels[topic_id] = f"{name} ({', '.join(top_keywords)})"
        else:
            custom_labels[topic_id] = name

    # Set custom labels
    topic_model.set_topic_labels(custom_labels)

    return topic_model

def create_topic_summary_report(topic_model, news_df, topic_names, original_categories=None):
    """
    Create comprehensive topic summary report with generated names

    Args:
        topic_model: BERTopic model with custom names
        news_df: DataFrame with topic assignments
        topic_names: Dictionary of generated topic names
        original_categories: Column name for original categories (optional)

    Returns:
        DataFrame with topic summary
    """
    print("Creating topic summary report...")

    topic_info = topic_model.get_topic_info()
    summary_data = []

    for _, row in topic_info.iterrows():
        topic_id = row['Topic']

        if topic_id == -1:  # Skip outliers
            continue

        # Basic info
        topic_docs = news_df[news_df['bertopic_topic'] == topic_id]

        summary = {
            'Topic_ID': topic_id,
            'Generated_Name': topic_names.get(topic_id, f"Topic {topic_id}"),
            'Document_Count': len(topic_docs),
            'Percentage': len(topic_docs) / len(news_df) * 100
        }

        # Keywords
        keywords = topic_model.get_topic(topic_id)
        if keywords:
            summary['Top_Keywords'] = ', '.join([word for word, score in keywords[:5]])
            summary['Keyword_Scores'] = ', '.join([f"{score:.3f}" for word, score in keywords[:5]])

        # Cluster information if available
        if 'cluster' in news_df.columns:
            clusters_involved = topic_docs['cluster'].nunique()
            dominant_cluster = topic_docs['cluster'].mode().iloc[0] if len(topic_docs) > 0 else -1
            cluster_alignment = (topic_docs['cluster'] == dominant_cluster).mean() * 100 if len(topic_docs) > 0 else 0

            summary['Clusters_Involved'] = clusters_involved
            summary['Dominant_Cluster'] = dominant_cluster
            summary['Cluster_Alignment_Pct'] = cluster_alignment

        # Original category alignment if available
        if original_categories and original_categories in news_df.columns:
            if len(topic_docs) > 0:
                category_counts = topic_docs[original_categories].value_counts()
                dominant_category = category_counts.index[0]
                category_alignment = category_counts.iloc[0] / len(topic_docs) * 100

                summary['Dominant_Original_Category'] = dominant_category
                summary['Category_Alignment_Pct'] = category_alignment

        # Sample texts
        if len(topic_docs) > 0:
            sample_text = topic_docs['text'].iloc[0][:200] + "..."
            summary['Sample_Text'] = sample_text

        summary_data.append(summary)

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Document_Count', ascending=False)

    return summary_df

def visualize_named_topics(topic_model, news_df, topic_names, top_n=15):
    """
    Create visualizations using the generated topic names
    """
    print("Creating visualizations with named topics...")

    # Get topic info
    topic_info = topic_model.get_topic_info()
    topic_info_filtered = topic_info[topic_info['Topic'] != -1].head(top_n)

    plt.figure(figsize=(20, 12))

    # 1. Topic distribution with names
    plt.subplot(2, 2, 1)
    topic_ids = topic_info_filtered['Topic'].tolist()
    topic_labels = [topic_names.get(tid, f"Topic {tid}") for tid in topic_ids]
    counts = topic_info_filtered['Count'].tolist()

    bars = plt.barh(range(len(topic_labels)), counts)
    plt.yticks(range(len(topic_labels)), topic_labels)
    plt.xlabel('Number of Documents')
    plt.title(f'Top {top_n} Topics by Document Count\n(Generated Names)')
    plt.gca().invert_yaxis()

    # Color bars
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 2. Topic distribution pie chart
    plt.subplot(2, 2, 2)
    if len(topic_labels) > 8:
        # Group smaller topics
        pie_labels = topic_labels[:8] + ['Others']
        pie_sizes = counts[:8] + [sum(counts[8:])]
    else:
        pie_labels = topic_labels
        pie_sizes = counts

    plt.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Topic Distribution\n(Named Topics)')

    # 3. Category alignment (if available)
    if 'category' in news_df.columns:
        plt.subplot(2, 2, 3)

        alignment_data = []
        alignment_labels = []

        for topic_id in topic_ids[:10]:  # Top 10 topics
            topic_docs = news_df[news_df['bertopic_topic'] == topic_id]
            if len(topic_docs) > 0:
                dominant_category = topic_docs['category'].mode().iloc[0]
                alignment_pct = (topic_docs['category'] == dominant_category).mean() * 100
                alignment_data.append(alignment_pct)
                alignment_labels.append(topic_names.get(topic_id, f"Topic {topic_id}"))

        bars = plt.barh(range(len(alignment_labels)), alignment_data)
        plt.yticks(range(len(alignment_labels)), alignment_labels)
        plt.xlabel('Category Alignment Percentage')
        plt.title('Topic-Category Alignment\n(Higher = More Coherent)')
        plt.gca().invert_yaxis()

        # Color code by alignment strength
        colors = plt.cm.RdYlGn(np.array(alignment_data) / 100)
        for bar, color in zip(bars, colors):
            bar.set_color(color)

    # 4. Topic sizes vs coherence
    plt.subplot(2, 2, 4)
    sizes = []
    coherence_scores = []
    labels = []

    for topic_id in topic_ids[:15]:
        topic_docs = news_df[news_df['bertopic_topic'] == topic_id]
        size = len(topic_docs)

        # Simple coherence metric: category alignment
        if 'category' in news_df.columns and len(topic_docs) > 0:
            dominant_category = topic_docs['category'].mode().iloc[0]
            coherence = (topic_docs['category'] == dominant_category).mean() * 100
        else:
            coherence = 50  # Default

        sizes.append(size)
        coherence_scores.append(coherence)
        labels.append(topic_names.get(topic_id, f"Topic {topic_id}"))

    scatter = plt.scatter(sizes, coherence_scores, s=100, alpha=0.7, c=range(len(sizes)), cmap='viridis')
    plt.xlabel('Topic Size (Number of Documents)')
    plt.ylabel('Topic Coherence (%)')
    plt.title('Topic Size vs Coherence\n(Named Topics)')
    plt.grid(True, alpha=0.3)

    # Add labels for some points
    for i, label in enumerate(labels[:8]):  # Label top 8 topics
        plt.annotate(label, (sizes[i], coherence_scores[i]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, alpha=0.8)

    plt.tight_layout()
    plt.show()

def save_topic_results(topic_summary_df, topic_names, output_path="topic_results.xlsx"):
    """
    Save topic modeling results to Excel file
    """
    print(f"Saving results to {output_path}...")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Topic summary
        topic_summary_df.to_excel(writer, sheet_name='Topic_Summary', index=False)

        # Topic names mapping
        names_df = pd.DataFrame(list(topic_names.items()), columns=['Topic_ID', 'Generated_Name'])
        names_df.to_excel(writer, sheet_name='Topic_Names', index=False)

    print(f"Results saved to {output_path}")

def enhanced_topic_modeling_with_ollama(topic_model, news_df, model_name="llama3.2:3b"):
    """
    Enhanced topic modeling pipeline with Ollama integration

    Args:
        topic_model: Fitted BERTopic model
        news_df: DataFrame with topic assignments
        model_name: Ollama model name to use

    Returns:
        Dictionary with all results
    """
    print("\n" + "="*60)
    print("ENHANCED TOPIC MODELING WITH OLLAMA")
    print("="*60)

    # Step 1: Generate topic names with Ollama
    topic_names = batch_generate_topic_names(
        topic_model, news_df, model_name,
        exclude_outliers=True, use_sample_texts=True
    )

    # Step 2: Update topic model with generated names
    topic_model_updated = update_topic_model_with_names(topic_model, topic_names)

    # Step 3: Create comprehensive summary report
    topic_summary_df = create_topic_summary_report(
        topic_model_updated, news_df, topic_names,
        original_categories='category'
    )

    # Step 4: Display summary
    print("\n" + "="*50)
    print("GENERATED TOPIC NAMES SUMMARY")
    print("="*50)

    for topic_id, name in sorted(topic_names.items()):
        topic_docs = news_df[news_df['bertopic_topic'] == topic_id]
        keywords = topic_model.get_topic(topic_id)
        keyword_str = ', '.join([word for word, score in keywords[:5]]) if keywords else 'N/A'

        print(f"\nTopic {topic_id}: {name}")
        print(f"  Documents: {len(topic_docs)}")
        print(f"  Keywords: {keyword_str}")

    # Step 5: Create enhanced visualizations
    visualize_named_topics(topic_model_updated, news_df, topic_names, top_n=15)

    # Step 6: Save results
    save_topic_results(topic_summary_df, topic_names)

    # Step 7: Print detailed summary table
    print("\n" + "="*80)
    print("DETAILED TOPIC SUMMARY")
    print("="*80)

    display_columns = ['Topic_ID', 'Generated_Name', 'Document_Count', 'Percentage', 'Top_Keywords']
    if 'Category_Alignment_Pct' in topic_summary_df.columns:
        display_columns.append('Category_Alignment_Pct')

    print(topic_summary_df[display_columns].to_string(index=False))

    return {
        'topic_names': topic_names,
        'topic_model_updated': topic_model_updated,
        'topic_summary_df': topic_summary_df,
        'enhanced_topic_info': topic_model_updated.get_topic_info()
    }

def visualize_bertopic_comprehensive(topic_model, news_df, top_n_topics=20):
    """
    Comprehensive visualization suite for BERTopic results

    Args:
        topic_model: Fitted BERTopic model
        news_df: DataFrame with BERTopic results
        top_n_topics: Number of top topics to display
    """

    print("Creating comprehensive BERTopic visualizations...")

    # 1. Topic Overview Bar Chart
    plt.figure(figsize=(15, 10))

    # Get topic info
    topic_info = topic_model.get_topic_info()

    # Filter out topic -1 (outliers) and get top N topics
    topic_info_filtered = topic_info[topic_info['Topic'] != -1].head(top_n_topics)

    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(topic_info_filtered)), topic_info_filtered['Count'])
    plt.title(f'Top {top_n_topics} Topics by Document Count')
    plt.xlabel('Topic Rank')
    plt.ylabel('Number of Documents')
    plt.xticks(range(len(topic_info_filtered)),
               [f"Topic {t}" for t in topic_info_filtered['Topic']],
               rotation=45)

    # Color bars by frequency
    colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 2. Topic Distribution Pie Chart
    plt.subplot(2, 2, 2)
    # Group smaller topics together
    sizes = topic_info_filtered['Count'].tolist()
    labels = [f"Topic {t}" for t in topic_info_filtered['Topic']]

    # If there are many small topics, group them
    if len(sizes) > 10:
        other_count = sum(sizes[10:])
        sizes = sizes[:10] + [other_count]
        labels = labels[:10] + ['Others']

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Topic Distribution')

    # 3. Outliers vs Valid Topics
    plt.subplot(2, 2, 3)
    outliers = len(news_df[news_df['bertopic_topic'] == -1])
    valid_topics = len(news_df[news_df['bertopic_topic'] != -1])

    plt.bar(['Valid Topics', 'Outliers'], [valid_topics, outliers],
            color=['skyblue', 'lightcoral'])
    plt.title('Outliers vs Valid Topic Assignments')
    plt.ylabel('Number of Documents')

    # Add percentage labels
    total = valid_topics + outliers
    plt.text(0, valid_topics/2, f'{valid_topics/total*100:.1f}%',
             ha='center', va='center', fontweight='bold')
    plt.text(1, outliers/2, f'{outliers/total*100:.1f}%',
             ha='center', va='center', fontweight='bold')

    # 4. Topic Keywords Heatmap (simplified)
    plt.subplot(2, 2, 4)
    # Get top keywords for top topics
    top_topics = topic_info_filtered['Topic'].head(10).tolist()
    keyword_matrix = []
    topic_labels = []

    for topic_id in top_topics:
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            # Get top 5 words and their scores
            words, scores = zip(*topic_words[:3])
            keyword_matrix.append(scores)
            topic_labels.append(f"T{topic_id}")

    if keyword_matrix:
        keyword_matrix = np.array(keyword_matrix)
        sns.heatmap(keyword_matrix,
                    yticklabels=topic_labels,
                    xticklabels=[f"Word {i+1}" for i in range(keyword_matrix.shape[1])],
                    annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Topic-Keyword Strength Heatmap')
        plt.xlabel('Top Keywords (by rank)')
        plt.ylabel('Topics')

    plt.tight_layout()
    plt.show()

def create_topic_wordclouds(topic_model, top_n_topics=12):
    """
    Create word clouds for top topics
    """
    topic_info = topic_model.get_topic_info()
    topic_info_filtered = topic_info[topic_info['Topic'] != -1].head(top_n_topics)

    # Calculate grid dimensions
    n_cols = 4
    n_rows = (top_n_topics + n_cols - 1) // n_cols

    plt.figure(figsize=(20, 5 * n_rows))

    for idx, (_, row) in enumerate(topic_info_filtered.iterrows()):
        topic_id = row['Topic']
        topic_words = topic_model.get_topic(topic_id)

        if topic_words:
            # Create word frequency dictionary
            word_freq = {word: score for word, score in topic_words}

            # Generate word cloud
            wordcloud = WordCloud(width=400, height=300,
                                  background_color='white',
                                  colormap='viridis',
                                  max_words=50).generate_from_frequencies(word_freq)

            plt.subplot(n_rows, n_cols, idx + 1)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'Topic {topic_id} ({row["Count"]} docs)', fontsize=12, fontweight='bold')
            plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_topic_keywords_detailed(topic_model, top_n_topics=10, top_n_words=10):
    """
    Detailed visualization of topic keywords with scores
    """
    topic_info = topic_model.get_topic_info()
    topic_info_filtered = topic_info[topic_info['Topic'] != -1].head(top_n_topics)

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(topic_info_filtered.iterrows()):
        if idx >= 10:  # Limit to 10 subplots
            break

        topic_id = row['Topic']
        topic_words = topic_model.get_topic(topic_id)

        if topic_words:
            words, scores = zip(*topic_words[:top_n_words])

            # Create horizontal bar chart
            y_pos = np.arange(len(words))
            axes[idx].barh(y_pos, scores, color=plt.cm.Set3(idx))
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(words)
            axes[idx].invert_yaxis()
            axes[idx].set_xlabel('Relevance Score')
            axes[idx].set_title(f'Topic {topic_id}\n({row["Count"]} documents)',
                                fontsize=10, fontweight='bold')
            axes[idx].grid(axis='x', alpha=0.3)

    # Hide empty subplots
    for idx in range(len(topic_info_filtered), 10):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()

def analyze_cluster_topic_relationship(news_df, topic_model):
    """
    Analyze the relationship between HDBSCAN clusters and BERTopic topics
    """
    print("Analyzing Cluster-Topic Relationships...")

    # Create cross-tabulation
    cluster_topic_crosstab = pd.crosstab(news_df['cluster'],
                                         news_df['bertopic_topic'],
                                         margins=True)

    print("\nCluster-Topic Cross-tabulation:")
    print(cluster_topic_crosstab)

    # Calculate alignment metrics
    # For each cluster, find the dominant topic
    cluster_topic_alignment = {}

    for cluster_id in news_df['cluster'].unique():
        if cluster_id == -1:  # Skip noise points
            continue

        cluster_data = news_df[news_df['cluster'] == cluster_id]
        topic_counts = cluster_data['bertopic_topic'].value_counts()

        if len(topic_counts) > 0:
            dominant_topic = topic_counts.index[0]
            dominant_topic_pct = topic_counts.iloc[0] / len(cluster_data) * 100

            cluster_topic_alignment[cluster_id] = {
                'dominant_topic': dominant_topic,
                'alignment_percentage': dominant_topic_pct,
                'total_docs': len(cluster_data),
                'topic_distribution': topic_counts.head(3).to_dict()
            }

    # Display alignment results
    print("\nCluster-Topic Alignment Analysis:")
    print("=" * 60)
    for cluster_id, info in cluster_topic_alignment.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Documents: {info['total_docs']}")
        print(f"  Dominant Topic: {info['dominant_topic']}")
        print(f"  Alignment: {info['alignment_percentage']:.1f}%")
        print(f"  Top 3 Topics: {info['topic_distribution']}")

    return cluster_topic_alignment


def create_interactive_topic_visualization(topic_model, news_df):
    """
    Create interactive visualizations using Plotly
    """
    try:
        # Use BERTopic's built-in visualization if available
        fig = topic_model.visualize_topics()
        fig.show()

        # Create interactive topic distribution
        topic_info = topic_model.get_topic_info()
        topic_info_filtered = topic_info[topic_info['Topic'] != -1].head(20)

        fig = px.bar(topic_info_filtered,
                     x='Topic',
                     y='Count',
                     title='Interactive Topic Distribution',
                     hover_data=['Name'])
        fig.update_layout(xaxis_title="Topic ID", yaxis_title="Number of Documents")
        fig.show()

    except Exception as e:
        print(f"Interactive visualization failed: {e}")
        print("Make sure you have plotly installed: pip install plotly")

def comprehensive_topic_analysis(topic_model, news_df, original_categories=None):
    """
    Complete analysis pipeline combining all visualizations and analyses
    """
    print("Starting Comprehensive Topic Analysis...")
    print("=" * 60)

    # 1. Basic topic statistics
    topic_info = topic_model.get_topic_info()
    print(f"Total topics discovered: {len(topic_info) - 1}")  # -1 for outlier topic
    print(f"Documents assigned to topics: {len(news_df[news_df['bertopic_topic'] != -1])}")
    print(f"Outlier documents: {len(news_df[news_df['bertopic_topic'] == -1])}")

    # 2. Main visualizations
    visualize_bertopic_comprehensive(topic_model, news_df, top_n_topics=15)

    # 3. Word clouds
    create_topic_wordclouds(topic_model, top_n_topics=12)

    # 4. Detailed keyword analysis
    visualize_topic_keywords_detailed(topic_model, top_n_topics=10)

    # 5. Cluster-topic relationship analysis
    cluster_topic_alignment = analyze_cluster_topic_relationship(news_df, topic_model)

    # 7. Interactive visualizations (if plotly is available)
    create_interactive_topic_visualization(topic_model, news_df)

    # 8. Compare with original categories if available
    if original_categories is not None:
        compare_topics_with_categories(topic_model, news_df, original_categories)

    return {
        'topic_info': topic_info,
        'cluster_topic_alignment': cluster_topic_alignment
    }

def compare_topics_with_categories(topic_model, news_df, category_column='category'):
    """
    Compare BERTopic results with original categories
    """
    print("\nComparing BERTopic Topics with Original Categories...")
    print("=" * 60)

    # Create category-topic cross-tabulation
    category_topic_crosstab = pd.crosstab(news_df[category_column],
                                          news_df['bertopic_topic'],
                                          margins=True)

    print("Category-Topic Cross-tabulation (top 10 categories):")
    print(category_topic_crosstab.head(10))

    # Find the most representative topic for each category
    category_topic_mapping = {}
    for category in news_df[category_column].unique():
        category_data = news_df[news_df[category_column] == category]
        topic_counts = category_data['bertopic_topic'].value_counts()

        if len(topic_counts) > 0 and topic_counts.index[0] != -1:
            dominant_topic = topic_counts.index[0]
            dominant_topic_pct = topic_counts.iloc[0] / len(category_data) * 100

            # Get topic words
            topic_words = topic_model.get_topic(dominant_topic)
            top_words = [word for word, _ in topic_words[:3]] if topic_words else []

            category_topic_mapping[category] = {
                'dominant_topic': dominant_topic,
                'alignment_percentage': dominant_topic_pct,
                'top_words': top_words,
                'total_docs': len(category_data)
            }

    # Display mapping
    print("\nCategory-Topic Mapping:")
    for category, info in sorted(category_topic_mapping.items(),
                                 key=lambda x: x[1]['alignment_percentage'],
                                 reverse=True):
        print(f"\n{category}:")
        print(f"  → Topic {info['dominant_topic']} ({info['alignment_percentage']:.1f}% alignment)")
        print(f"  → Keywords: {', '.join(info['top_words'])}")
        print(f"  → Documents: {info['total_docs']}")

# data load
def load_news_dataset(json_file_path):
    """
    Load the Kaggle News Category Classification dataset from JSON file
    and extract 'short_description' and 'headline' columns.

    Args:
        json_file_path (str): Path to the JSON file

    Returns:
        pd.DataFrame: DataFrame containing 'headline' and 'short_description' columns
    """
    # Read the JSON file line by line (each line is a separate JSON object)
    data = []
    with open(json_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue  # Skip malformed lines

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Ensure required columns exist
    if 'short_description' not in df.columns or 'headline' not in df.columns:
        raise ValueError("Required columns 'short_description' and 'headline' not found in dataset")

    # Handle missing values
    df['short_description'] = df['short_description'].fillna('')
    df['headline'] = df['headline'].fillna('')

    # Combine text fields
    df['text'] = df['short_description'] + ' ' + df['headline']

    # Extract only the required columns
    final_df = df[['text', 'category']].copy()

    # Remove empty texts
    final_df = final_df[final_df['text'].str.strip() != '']

    return final_df

def plot_reduced_embeddings(news_df, n_components=10):
    # Plot the first two UMAP components
    plt.figure(figsize=(10, 8))
    for category in news_df['category'].unique():
        subset = news_df[news_df['category'] == category]
        plt.scatter(subset['umap_1'], subset['umap_2'], label=category, alpha=0.6)

    plt.title('UMAP Projection of News Embeddings')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def evaluate_clustering_comprehensive(X, labels, true_labels=None, label_names=None):
    """
    Comprehensive clustering evaluation with multiple metrics

    Args:
        X: Feature matrix (embeddings)
        labels: Cluster labels from algorithm
        true_labels: Ground truth labels (optional)
        label_names: Names for true labels (optional)

    Returns:
        Dictionary containing all evaluation metrics
    """
    metrics = {}

    # Handle noise points (label -1)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)

    metrics['n_clusters'] = n_clusters
    metrics['n_noise'] = n_noise
    metrics['noise_ratio'] = n_noise / len(labels)

    # Internal validation metrics (only for non-noise points)
    if n_clusters > 1:
        # Remove noise points for internal metrics
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 0:
            X_clean = X[non_noise_mask]
            labels_clean = labels[non_noise_mask]


            # Silhouette Score: Measures how similar a data point is to its own cluster compared to other clusters
            # Scores range from -1 (bad clustering) to +1 (dense, well-separated clusters)
            # Calinski-Harabasz Index: Measures the ratio of between-cluster variance to within-cluster variance. A higher score is better.
            # Davies-Bouldin Index: Measures the average similarity between clusters. A lower score is better.
            if len(np.unique(labels_clean)) > 1:
                try:
                    metrics['silhouette'] = silhouette_score(X_clean, labels_clean)
                    metrics['calinski_harabasz'] = calinski_harabasz_score(X_clean, labels_clean)
                    metrics['davies_bouldin'] = davies_bouldin_score(X_clean, labels_clean)
                except:
                    metrics['silhouette'] = -1
                    metrics['calinski_harabasz'] = 0
                    metrics['davies_bouldin'] = float('inf')

    # External validation metrics (if ground truth is available)
    # Adjusted Rand Index (ARI): Measures the similarity between two clusterings, correcting for chance. A score of 1.0 means a perfect match.
    # Normalized Mutual Information (NMI): Measures the mutual dependence between two clusterings. A score of 1.0 indicates a perfect correlation.
    # Homogeneity, Completeness, V-Measure: These three metrics evaluate if each cluster contains only members of a single class (homogeneity), if all members of a given class are assigned to the same cluster (completeness), and the harmonic mean of the two (V-Measure).
    if true_labels is not None:
        try:
            metrics['adjusted_rand_index'] = adjusted_rand_score(true_labels, labels)
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, labels)
            metrics['homogeneity'] = homogeneity_score(true_labels, labels)
            metrics['completeness'] = completeness_score(true_labels, labels)
            metrics['v_measure'] = v_measure_score(true_labels, labels)
        except:
            metrics['adjusted_rand_index'] = 0
            metrics['normalized_mutual_info'] = 0
            metrics['homogeneity'] = 0
            metrics['completeness'] = 0
            metrics['v_measure'] = 0

    return metrics

def efficient_hdbscan_tuning(embeddings, true_labels=None, sample_size=None):
    """
    Efficient HDBSCAN parameter tuning with targeted parameter combinations

    Args:
        embeddings: Feature matrix (embeddings)
        true_labels: Ground truth labels for external validation (optional)
        sample_size: Sample size for faster tuning (optional)

    Returns:
        best_params, best_score, all_results
    """

    print("Starting efficient HDBSCAN parameter tuning...")

    # Sample data if requested for faster tuning
    if sample_size and len(embeddings) > sample_size:
        print(f"Sampling {sample_size} points from {len(embeddings)} for parameter tuning...")
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings_sample = embeddings[indices]
        true_labels_sample = true_labels[indices] if true_labels is not None else None
    else:
        embeddings_sample = embeddings
        true_labels_sample = true_labels

    # scaled embeddings
    embeddings_scaled = embeddings_sample

    # Define targeted parameter combinations (much more efficient than grid search)
    param_combinations = [
        # Small, tight clusters
        {'min_cluster_size': 15, 'min_samples': 5, 'cluster_selection_epsilon': 0.0},
        {'min_cluster_size': 20, 'min_samples': 5, 'cluster_selection_epsilon': 0.0},

        # Medium clusters with some flexibility
        {'min_cluster_size': 25, 'min_samples': 10, 'cluster_selection_epsilon': 0.1},
        {'min_cluster_size': 30, 'min_samples': 10, 'cluster_selection_epsilon': 0.1},
        {'min_cluster_size': 35, 'min_samples': 15, 'cluster_selection_epsilon': 0.1},

        # Larger, more flexible clusters
        {'min_cluster_size': 40, 'min_samples': 15, 'cluster_selection_epsilon': 0.2},
        {'min_cluster_size': 50, 'min_samples': 20, 'cluster_selection_epsilon': 0.2},
        {'min_cluster_size': 60, 'min_samples': 25, 'cluster_selection_epsilon': 0.3},

        # Very large clusters (for datasets with clear large groups)
        {'min_cluster_size': 80, 'min_samples': 30, 'cluster_selection_epsilon': 0.4},
        {'min_cluster_size': 100, 'min_samples': 40, 'cluster_selection_epsilon': 0.5},
    ]

    # Additional parameters to test
    additional_params = {
        'metric': 'euclidean',
        'alpha': 1.0,
        'algorithm': 'best',
        'leaf_size': 40,
        'cluster_selection_method':"eom"
    }

    results = []
    best_score = -1
    best_params = None

    print(f"Testing {len(param_combinations)} parameter combinations...")

    for i, params in enumerate(param_combinations):
        start_time = time.time()

        # Combine with additional parameters
        full_params = {**params, **additional_params}

        try:
            # Fit HDBSCAN
            clusterer = hdbscan.HDBSCAN(**full_params)
            cluster_labels = clusterer.fit_predict(embeddings_scaled)

            # Evaluate clustering
            metrics = evaluate_clustering_comprehensive(
                embeddings_scaled, cluster_labels, true_labels_sample
            )

            # Calculate composite score (you can adjust weights based on your priorities)
            if true_labels_sample is not None:
                # If we have ground truth, prioritize external validation
                composite_score = (
                        0.3 * metrics.get('silhouette', 0) +
                        0.3 * metrics.get('adjusted_rand_index', 0) +
                        0.2 * metrics.get('normalized_mutual_info', 0) +
                        0.1 * metrics.get('v_measure', 0) +
                        0.1 * (1 - metrics.get('noise_ratio', 1))  # Lower noise is better
                )
            else:
                # If no ground truth, focus on internal metrics
                composite_score = (
                        0.4 * metrics.get('silhouette', 0) +
                        0.3 * (metrics.get('calinski_harabasz', 0) / 1000) +  # Normalize CH score
                        0.2 * (1 - metrics.get('davies_bouldin', float('inf')) / 10) +  # Lower DB is better
                        0.1 * (1 - metrics.get('noise_ratio', 1))  # Lower noise is better
                )

            # Store results
            result = {
                'params': full_params,
                'metrics': metrics,
                'composite_score': composite_score,
                'time': time.time() - start_time
            }
            results.append(result)

            # Update best score
            if composite_score > best_score:
                best_score = composite_score
                best_params = full_params

            # Print progress
            print(f"  {i+1:2d}/{len(param_combinations)}: "
                  f"Score={composite_score:.4f}, "
                  f"Clusters={metrics['n_clusters']}, "
                  f"Noise={metrics['noise_ratio']:.3f}, "
                  f"Time={time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"  {i+1:2d}/{len(param_combinations)}: Failed - {str(e)}")
            continue

    # Sort results by composite score
    results.sort(key=lambda x: x['composite_score'], reverse=True)

    print(f"\nBest parameters found:")
    print(f"Composite Score: {best_score:.4f}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    return best_params, best_score, results

def display_tuning_results(results, top_n=5):
    """Display top N results from parameter tuning"""

    print(f"\nTop {top_n} Parameter Combinations:")
    print("=" * 80)

    for i, result in enumerate(results[:top_n]):
        print(f"\nRank {i+1}:")
        print(f"Composite Score: {result['composite_score']:.4f}")
        print(f"Parameters: {result['params']}")
        print("Metrics:")

        metrics = result['metrics']
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

def plot_parameter_analysis(results):
    """Visualize parameter tuning results"""

    # Extract data for plotting
    scores = [r['composite_score'] for r in results]
    n_clusters = [r['metrics']['n_clusters'] for r in results]
    noise_ratios = [r['metrics']['noise_ratio'] for r in results]
    silhouette_scores = [r['metrics'].get('silhouette', 0) for r in results]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('HDBSCAN Parameter Tuning Analysis', fontsize=16)

    # 1. Composite scores
    axes[0, 0].plot(range(len(scores)), scores, 'bo-', alpha=0.7)
    axes[0, 0].set_title('Composite Scores by Parameter Combination')
    axes[0, 0].set_xlabel('Parameter Combination')
    axes[0, 0].set_ylabel('Composite Score')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Number of clusters vs Score
    axes[0, 1].scatter(n_clusters, scores, alpha=0.7, c=noise_ratios, cmap='viridis')
    axes[0, 1].set_title('Clusters vs Score (colored by noise ratio)')
    axes[0, 1].set_xlabel('Number of Clusters')
    axes[0, 1].set_ylabel('Composite Score')
    axes[0, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar.set_label('Noise Ratio')

    # 3. Noise ratio distribution
    axes[1, 0].hist(noise_ratios, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Distribution of Noise Ratios')
    axes[1, 0].set_xlabel('Noise Ratio')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Silhouette vs Composite Score
    axes[1, 1].scatter(silhouette_scores, scores, alpha=0.7)
    axes[1, 1].set_title('Silhouette Score vs Composite Score')
    axes[1, 1].set_xlabel('Silhouette Score')
    axes[1, 1].set_ylabel('Composite Score')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def final_clustering_with_best_params(embeddings, best_params):
    """
    Apply final clustering using the best parameters found

    Args:
        embeddings: Full embedding matrix
        best_params: Best parameters from tuning

    Returns:
        clusterer, cluster_labels
    """

    print("Applying final clustering with best parameters...")

    embeddings_scaled = embeddings

    # Apply clustering
    clusterer = hdbscan.HDBSCAN(**best_params)
    cluster_labels = clusterer.fit_predict(embeddings_scaled)

    # Print results
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(cluster_labels == -1)

    print(f"Final clustering results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.2f}%)")
    print(f"  Cluster sizes: {np.bincount(cluster_labels[cluster_labels >= 0])}")

    return clusterer, cluster_labels, embeddings_scaled

# clustering
def improved_clustering_pipeline(news_df, embeddings, sample_size_for_tuning=5000):
    """
    Complete improved clustering pipeline to replace your original clustering section

    Args:
        news_df: Your news dataframe
        embeddings: Embedding matrix (2D numpy array)
        sample_size_for_tuning: Sample size for parameter tuning (for speed)

    Returns:
        news_df with cluster labels, best_params, tuning_results
    """

    print("Starting improved clustering pipeline...")

    # Prepare ground truth labels for external validation
    true_labels = pd.Categorical(news_df['category']).codes

    # Step 1: Efficient parameter tuning
    best_params, best_score, tuning_results = efficient_hdbscan_tuning(
        embeddings,
        true_labels=true_labels,
        # sample_size=sample_size_for_tuning
    )

    # Step 2: Display tuning results
    display_tuning_results(tuning_results, top_n=5)

    # Step 3: Visualize parameter analysis
    plot_parameter_analysis(tuning_results)

    # Step 4: Apply final clustering with best parameters
    clusterer, cluster_labels, scaled_embeddings = final_clustering_with_best_params(
        embeddings, best_params
    )

    # Step 5: Add cluster labels to dataframe
    news_df['cluster'] = cluster_labels

    # Step 6: Final evaluation
    final_metrics = evaluate_clustering_comprehensive(
        scaled_embeddings, cluster_labels, true_labels
    )

    print("\nFinal Clustering Evaluation:")
    print("=" * 40)
    for metric, value in final_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    return news_df, best_params, tuning_results, final_metrics, clusterer


def plot_umap_clusters_single(news_df):
    """
    Create a single UMAP scatterplot showing clusters in 2D space
    """
    plt.figure(figsize=(10, 8))

    # UMAP colored by clusters
    unique_clusters = sorted(news_df['cluster'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = news_df[news_df['cluster'] == cluster_id]
        if cluster_id == -1:  # Noise points
            plt.scatter(cluster_data['umap_1'], cluster_data['umap_2'],
                        c='red', alpha=0.5, s=15, label=f'Noise ({len(cluster_data)})',
                        marker='x')
        else:
            plt.scatter(cluster_data['umap_1'], cluster_data['umap_2'],
                        c=[colors[i]], alpha=0.7, s=20,
                        label=f'Cluster {cluster_id} ({len(cluster_data)})')

    plt.title('UMAP Projection - Colored by Clusters', fontsize=16, fontweight='bold')
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# comprehensive pipeline for news article clustering. It takes a JSON dataset of news articles,
# preprocesses the text, converts it into numerical embeddings, and then performs clustering to group similar articles together.

def main():
    # Replace with your actual file path
    file_path = "News_Category_Dataset_v3.json"  # Update this path

    try:
        # Load the data
        print("Loading dataset...")
        news_df = load_news_dataset(file_path)

        # Sample for faster processing (remove for full dataset)
        news_df = news_df.sample(n=min(10000, len(news_df)), random_state=42)

        print(f"Loaded {len(news_df)} articles")
        print(f"Categories: {news_df['category'].nunique()}")
        print("\nSample data:")
        print(news_df.head())

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Load a pre-trained sentence transformer model
        model = SentenceTransformer('thenlper/gte-small') # all-mpnet-base-v2

        # Convert the short_desc + headline into an embedding
        news_df['embeddings'] = news_df['text'].apply(lambda x: model.encode(x))

        print(news_df[["embeddings"]].shape)

        # Shape of the first embedding (assuming all embeddings have the same length)
        print(np.shape(news_df['embeddings'].iloc[0]))

        # Shape of the entire embedding field as a 2D array
        embeddings = np.stack(news_df['embeddings'].values)
        print(embeddings.shape)

        # Display the first few rows
        print(news_df.head())
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    try:
        # reduce the dimensionality of the embeddings
        # Initialize UMAP with the desired parameters
        reducer = UMAP(
            n_components=20,
            min_dist=0.0,
            metric='cosine',
            random_state=123
        )

        """
        If your embeddings are stored in a pandas DataFrame column as df["embeddings"], 
        where each entry is a vector (e.g., a list or numpy array), you should first stack these into a 
        2D numpy array before applying UMAP. You do not need to loop row by row—UMAP expects the entire matrix at once 
        for efficiency and correctness
        
        """
        # Stack embeddings into a 2D numpy array
        embedding_matrix = np.vstack(news_df["embeddings"].values)

        # Perform dimensionality reduction
        reduced_embeddings = reducer.fit_transform(embedding_matrix)

        print(reduced_embeddings.shape[1])

        # Add reduced dimensions as new columns to the DataFrame
        for idx in range(reduced_embeddings.shape[1]):
            news_df[f'umap_{idx+1}'] = reduced_embeddings[:, idx]

        # Display the first few rows
        print(news_df.head())

        # Call the function to plot
        plot_reduced_embeddings(news_df)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    try:
        # Clustering
        # Standardize embeddings for clustering
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(reduced_embeddings)

        # Run improved clustering pipeline
        news_df, best_params, tuning_results, final_metrics,clusterer = improved_clustering_pipeline(
            news_df, scaled_embeddings, sample_size_for_tuning=3000
        )

        print("Clustering completed successfully!")
        print(f"best_params are {best_params}")
        print(f"final metrics are {final_metrics}")

        # Comprehensive visualizations
        print("\nCreating visualizations...")
        plot_umap_clusters_single(news_df)           # UMAP colored by clusters

    except Exception as e:
        print(f"Error is {e}")

    # topic modelling
    # Text → Embeddings (via LLM) → UMAP (reduce dimensions) → HDBSCAN (cluster) → Topics
    try:
        topic_model = BERTopic(embedding_model=model, umap_model=reducer, hdbscan_model=clusterer)
        topics, probabilities = topic_model.fit_transform(news_df['text'], reduced_embeddings)

        # Get the topic information and add it to our DataFrame
        topic_info = topic_model.get_topic_info()
        news_df['bertopic_topic'] = topics

        print("BERTopic clustering complete.")
        print(topic_info.head(10))
        print("\nSample data with BERTopic cluster labels:")
        print(news_df.head())

    except Exception as e:
        print(f"Error is {e}")

    # HDBSCAN groups documents with similar embeddings into clusters
    # BERTopic uses these same clusters to identify topics and extract keywords
    # Each HDBSCAN cluster typically corresponds to one BERTopic topic
    # BERTopic analyzes all documents in each topic/cluster
    # Uses class-based TF-IDF to find words that distinguish each topic
    # Keywords are ranked by their importance to that specific topic
    try:
        # Run comprehensive topic analysis
        analysis_results = comprehensive_topic_analysis(
            topic_model,
            news_df,
            original_categories='category'  # Use your original category column
        )

        print("Topic analysis completed successfully!")

        # Check how well HDBSCAN clusters align with BERTopic topics
        alignment_check = news_df.groupby('cluster')['bertopic_topic'].agg(['count', 'nunique', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else -1])
        print(alignment_check)

        # Analyze topic coherence and size
        topic_analysis = news_df.groupby('bertopic_topic').agg({
            'cluster': 'nunique',  # How many clusters contribute to this topic
            'text': 'count',       # How many documents in this topic
            'category': lambda x: x.value_counts().iloc[0] if len(x) > 0 else 'Unknown'  # Most common original category
        })

        print(topic_analysis)

        # Get keywords for each topic
        for topic_id in range(topic_model.get_topic_info().shape[0]):
            if topic_id != -1:  # Skip outlier topic
                keywords = topic_model.get_topic(topic_id)
                print(f"Topic {topic_id}: {[word for word, score in keywords[:10]]}")

        # Basic Statistics
        print("Cluster-Topic Summary:")
        print(f"Unique clusters: {news_df['cluster'].nunique()}")
        print(f"Unique topics: {news_df['bertopic_topic'].nunique()}")
        print(f"Outlier documents: {(news_df['bertopic_topic'] == -1).sum()}")

        # Alignment Analysis
        cluster_topic_crosstab = pd.crosstab(news_df['cluster'], news_df['bertopic_topic'])

        # Calculate alignment percentage for each cluster
        for cluster_id in news_df['cluster'].unique():
            if cluster_id != -1:
                cluster_docs = news_df[news_df['cluster'] == cluster_id]
                dominant_topic = cluster_docs['bertopic_topic'].mode().iloc[0]
                alignment_pct = (cluster_docs['bertopic_topic'] == dominant_topic).mean() * 100
                print(f"Cluster {cluster_id}: {alignment_pct:.1f}% aligned with Topic {dominant_topic}")

        # Topic Content Analysis
        # Analyze each topic's content
        for topic_id in news_df['bertopic_topic'].unique():
            if topic_id != -1:
                topic_docs = news_df[news_df['bertopic_topic'] == topic_id]

                print(f"\nTopic {topic_id}:")
                print(f"  Documents: {len(topic_docs)}")
                print(f"  Clusters involved: {topic_docs['cluster'].nunique()}")

                # Keywords
                keywords = topic_model.get_topic(topic_id)
                print(f"  Keywords: {[word for word, score in keywords[:5]]}")

                # Sample documents
                print(f"  Sample texts:")
                for text in topic_docs['text'].head(2):
                    print(f"    - {text[:100]}...")

        # Validation Against Original Categories
        # If you have original categories (news_df['category']), compare them with discovered topics:
        # Topic-Category alignment
        category_topic_map = {}
        for category in news_df['category'].unique():
            cat_docs = news_df[news_df['category'] == category]
            dominant_topic = cat_docs['bertopic_topic'].mode().iloc[0] if len(cat_docs) > 0 else -1
            alignment = (cat_docs['bertopic_topic'] == dominant_topic).mean() * 100

            category_topic_map[category] = {
                'dominant_topic': dominant_topic,
                'alignment_percentage': alignment,
                'document_count': len(cat_docs)
            }

        # Display results
        for category, info in sorted(category_topic_map.items(),
                                     key=lambda x: x[1]['alignment_percentage'],
                                     reverse=True):
            print(f"{category}: Topic {info['dominant_topic']} ({info['alignment_percentage']:.1f}% alignment)")


    except Exception as e:
        print(f"Error in topic analysis: {e}")
        import traceback
        traceback.print_exc()

    try:
        # ========================= ENHANCED TOPIC MODELING WITH OLLAMA =========================
        print("\n" + "="*80)
        print("STARTING ENHANCED TOPIC MODELING WITH OLLAMA INTEGRATION")
        print("="*80)

        # Run the enhanced topic modeling pipeline with Ollama
        ollama_results = enhanced_topic_modeling_with_ollama(
            topic_model,
            news_df,
            model_name="llama3.2:3b"  # Change this to your preferred Ollama model
        )

        # Extract results
        topic_names = ollama_results['topic_names']
        topic_model_updated = ollama_results['topic_model_updated']
        topic_summary_df = ollama_results['topic_summary_df']

        print("\n" + "="*60)
        print("OLLAMA TOPIC MODELING COMPLETED SUCCESSFULLY!")
        print("="*60)

        # Additional analysis with named topics
        print("\nAdditional Analysis with Named Topics:")

        # Show topic evolution/trends if timestamp data is available
        if 'date' in news_df.columns:
            print("Creating topic trends over time...")
            # This would require additional timestamp processing

        # Topic coherence analysis
        print("\nTopic Coherence Analysis:")
        for topic_id, name in sorted(topic_names.items(), key=lambda x: len(news_df[news_df['bertopic_topic'] == x[0]]), reverse=True)[:10]:
            topic_docs = news_df[news_df['bertopic_topic'] == topic_id]
            coherence = len(topic_docs)

            # Calculate category diversity (lower is more coherent)
            if 'category' in news_df.columns:
                category_diversity = len(topic_docs['category'].unique())
                print(f"Topic {topic_id} ({name}): {coherence} docs, {category_diversity} categories")

    except Exception as e:
        print(f"Error in enhanced topic modeling with Ollama: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    results = main()