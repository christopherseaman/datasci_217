# Demo 1: Merge Operations

## Overview
Practice database-style joins with customer, product, and purchase data.

## Learning Objectives
- Master `pd.merge()` with different join types
- Understand when to use each join type
- Handle real-world merge challenges

## Activities

### 1. Basic Merge Operations
- Inner join between purchases and customers
- Left join to keep all purchases
- Right join to keep all customers
- Outer join to see everything

### 2. Join Validation and Debugging
- Check row counts before and after merge
- Handle duplicate keys and validation

### 3. Multi-column Merges
- Merge on composite keys (store + date)
- Handle overlapping column names with suffixes

## Key Concepts
- **Inner Join**: Only matching records from both tables
- **Left Join**: All records from left table + matching from right
- **Right Join**: All records from right table + matching from left
- **Outer Join**: All records from both tables

## Common Pitfalls
- Wrong join type = lost data
- Use left join to keep all customers
- Check for duplicate keys before merging
