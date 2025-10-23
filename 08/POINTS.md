# Data Aggregation and Group Operations

## The Split-Apply-Combine Paradigm

Data aggregation is the process of summarizing and grouping data to extract meaningful insights. The split-apply-combine paradigm is the foundation of data aggregation - you split data into groups, apply a function to each group, and combine the results. This is the bread and butter of data analysis because real-world data is rarely organized exactly how you need it.

- The split-apply-combine paradigm is the foundation of data aggregation - split data into groups, apply functions, combine results
- This approach works for any question that starts with "what if we group by..." - it's the most common pattern in data analysis
- GroupBy operations return a new DataFrame without modifying the original - always assign the result to capture it
- The key insight: you're not just calculating statistics, you're restructuring your data to answer specific questions
- Real-world data is rarely organized how you need it - aggregation is how you reshape it for analysis

## Basic GroupBy Operations

GroupBy operations let you calculate statistics within groups defined by one or more columns. The syntax is simple but the applications are endless - from calculating department averages to finding the top performer in each category.

- df.groupby('column') creates a GroupBy object that represents grouped data - it doesn't calculate anything until you specify an aggregation
- Common aggregation functions: mean(), sum(), count(), size(), min(), max(), std(), var() - each answers different business questions
- count() excludes null values, size() includes them - choose based on whether nulls matter for your analysis
- Use .agg() with a dictionary to apply different functions to different columns - this is more flexible than chaining multiple operations
- Always check the shape of your result - unexpected row counts indicate data loss or accidental many-to-many joins

## Advanced GroupBy Operations

GroupBy operations go beyond simple aggregation to include transformation, filtering, and custom functions. These advanced operations let you create new columns, remove groups, and apply complex logic within each group.

### Transform Operations

Transform operations apply a function to each group and return a result with the same shape as the original data. This is perfect for creating new columns that contain group-level statistics.

- transform() returns a Series with the same index as the original data - perfect for adding group statistics as new columns
- df['group_mean'] = df.groupby('group')['value'].transform('mean') adds the group mean to every row
- transform() is essential for normalization - subtracting group means, dividing by group standard deviations
- Use transform() when you need group statistics but want to keep the original row structure
- Common use case: "add a column showing how each employee's salary compares to their department average"

### Filter Operations

Filter operations remove entire groups based on a condition. This is different from row-level filtering - you're deciding which groups to keep or discard based on group-level properties.

- filter() removes entire groups based on a condition - use it when you want to exclude groups that don't meet criteria
- lambda x: len(x) > n keeps groups with more than n rows - useful for removing small groups that might be outliers
- lambda x: x['col'].mean() > threshold keeps groups where the average exceeds a threshold
- Filter operations are powerful for data cleaning - removing departments with too few employees, products with insufficient sales history
- Always check how many groups you're removing - you might be accidentally excluding important data

### Apply Operations

Apply operations let you use custom functions on each group. This is the most flexible but also most complex option - you can implement any logic that works on a group of rows.

- apply() lets you use custom functions on each group - this is where GroupBy becomes truly powerful
- Custom functions can return Series, DataFrames, or scalars - the return type determines the final result structure
- Use apply() when you need complex logic that doesn't fit standard aggregation functions
- Common pattern: apply(lambda x: x.nlargest(2, 'column')) to get the top 2 items from each group
- Be careful with apply() performance - it's slower than built-in aggregation functions for large datasets

### Hierarchical Grouping

Hierarchical grouping creates multi-level group structures that can be reshaped and analyzed in different ways. This emerges naturally when you group by multiple columns and need to explore different levels of aggregation.

- Grouping by multiple columns creates hierarchical structure - each combination of grouping values becomes a group
- Hierarchical grouping automatically creates MultiIndex structures that can be confusing but are very powerful
- Use .unstack() to convert hierarchical results to wide format - this is often what you want for presentation
- Hierarchical grouping is perfect for "drill-down" analysis - start with high-level groups, then explore subgroups
- Most people immediately use reset_index() to flatten hierarchical results - that's completely fine for most use cases

# LIVE DEMO!

# Pivot Tables and Cross-Tabulations

Pivot tables are powerful tools for summarizing and analyzing data across multiple dimensions. They're like Excel pivot tables but with the full power of pandas - you can aggregate, reshape, and analyze data in ways that would take dozens of lines of code manually.

- Pivot tables reshape data from long format to wide format by spreading one column's values across multiple columns
- pd.pivot_table() is more flexible than regular pivot() because it handles duplicates through aggregation
- The key parameters: values (what to aggregate), index (rows), columns (columns), aggfunc (how to aggregate)
- Pivot tables are perfect for "show me sales by product and region" type questions
- Always specify aggfunc explicitly - the default is mean, but you might want sum, count, or other functions

## Basic Pivot Tables

Basic pivot tables transform long-format data into wide-format summaries. This is the most common use case - you have data with multiple categories and want to see how values vary across those categories.

- pd.pivot_table(df, values='sales', index='product', columns='region') creates a product-by-region sales table
- Use aggfunc='sum' for totals, aggfunc='mean' for averages, aggfunc='count' for frequencies
- fill_value=0 replaces missing combinations with zeros - useful when not all combinations exist in your data
- margins=True adds row and column totals - helpful for understanding the overall picture
- Pivot tables automatically handle missing combinations by creating NaN values - use fill_value to control this

## Advanced Pivot Operations

Advanced pivot operations handle complex scenarios like multiple aggregation functions, hierarchical grouping, and missing value handling. These features make pivot tables suitable for sophisticated business analysis.

- Use aggfunc=['sum', 'mean'] to apply multiple aggregation functions - this creates hierarchical column structure
- margins=True with margins_name='Total' adds row and column totals with custom labels
- fill_value=0 replaces missing combinations with zeros instead of NaN - crucial for financial data
- dropna=False keeps missing combinations as NaN - useful when you want to see which combinations don't exist
- observed=True includes all category combinations, even those with no data - important for categorical data

# LIVE DEMO!

# Remote Computing and SSH

When your data is too big for your laptop, it's time to think about remote computing. SSH provides secure access to powerful remote servers that can handle massive datasets and long-running analyses.

- SSH (Secure Shell) provides encrypted remote access to servers - essential for working with large datasets
- ssh-keygen creates public/private key pairs for passwordless authentication - much more secure than passwords
- ssh-copy-id copies your public key to the server, enabling seamless connections
- scp (secure copy) transfers files between local and remote machines - essential for data movement
- Remote computing lets you leverage powerful servers for analysis that would be impossible locally

## SSH Fundamentals

SSH provides secure remote access to servers. Understanding the basics of SSH authentication and file transfer is essential for remote computing workflows.

- ssh username@hostname connects to a remote server - you'll be prompted for a password unless you've set up key authentication
- ssh-keygen -t rsa -b 4096 creates a strong RSA key pair for authentication
- ssh-copy-id username@hostname copies your public key to the server, enabling passwordless login
- scp file username@hostname:path copies files to the server
- scp username@hostname:file path copies files from the server to your local machine

## Remote Data Analysis

Remote data analysis requires setting up your computing environment on the server and managing data transfer between local and remote systems.

- Start Jupyter notebook on the server with --ip=0.0.0.0 to allow remote connections
- Use SSH port forwarding (ssh -L 8888:localhost:8888) to access Jupyter from your local browser
- Always use tmux or screen for long-running analysis - this prevents losing work when network connections drop
- Transfer data efficiently using scp or rsync for large files
- Keep your analysis code in version control so you can easily deploy it to remote servers

## Screen and tmux for Persistent Sessions

Persistent terminal sessions are essential for remote computing. They allow your analysis to continue running even when you disconnect from the server.

- tmux creates persistent terminal sessions that survive disconnections and network interruptions
- tmux new-session -s analysis creates a named session that's easy to identify and reconnect to
- Ctrl+b d detaches from a session while keeping it running in the background
- tmux attach-session -t analysis reconnects to a running session
- This is crucial for long-running analysis that might take hours or days to complete

# Performance Optimization

When working with large datasets, performance optimization can mean the difference between a 5-minute analysis and a 5-hour analysis. Understanding how to optimize GroupBy operations and memory usage is essential for real-world data analysis.

# FIXME: Add performance comparison chart showing groupby vs pivot_table vs manual aggregation speed

# FIXME: Add memory usage visualization for different aggregation methods

## Efficient GroupBy Operations

Efficient GroupBy operations require understanding how pandas processes grouped data and optimizing your code accordingly.

- Use categorical data types for grouping columns - this can dramatically improve performance for repeated grouping operations
- Avoid unnecessary data copying - work with views when possible, use inplace operations when appropriate
- Use specific aggregation functions rather than generic apply() when possible - built-in functions are much faster
- Consider using Dask or other parallel processing libraries for datasets that don't fit in memory
- Profile your code to identify bottlenecks - don't optimize blindly without measuring actual performance

## Parallel Processing

Parallel processing can significantly speed up GroupBy operations on large datasets by distributing the work across multiple CPU cores.

- Use multiprocessing.Pool to parallelize GroupBy operations across multiple CPU cores
- Split your data into chunks and process each chunk independently, then combine the results
- Be careful with memory usage - parallel processing can increase memory requirements significantly
- Consider using specialized libraries like Dask for out-of-core parallel processing
- Always measure performance improvements - parallel processing isn't always faster due to overhead

# LIVE DEMO!

# Key Takeaways

Data aggregation is a fundamental skill for any data scientist. The key is understanding the split-apply-combine paradigm and choosing the right tools for your specific analysis needs.

- Master the split-apply-combine paradigm - it's the foundation of data aggregation and answers most "what if we group by..." questions
- Use pivot tables for multi-dimensional analysis - they're more powerful than Excel and handle complex scenarios
- Leverage remote computing for large datasets - SSH and tmux make powerful servers accessible
- Optimize performance with efficient operations and parallel processing when needed
- Use persistent sessions for long-running analysis - tmux ensures your work survives disconnections
- Understand hierarchical grouping for complex data structures - but don't be afraid to flatten results when needed

You now have the skills to aggregate and summarize data effectively, even with large datasets that require remote computing resources.
