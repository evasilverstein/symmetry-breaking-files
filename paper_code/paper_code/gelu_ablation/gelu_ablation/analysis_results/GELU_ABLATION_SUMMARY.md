# GELU Ablation Study - Combined Results Summary

This file contains combined training validation loss and logic puzzle evaluation results.

## Overview

| Model | Training Val Loss | Logic Puzzle Loss | Δ |
|-------|------------------|-------------------|---|
| adam-124m-gelu-bQbV-seed42 | 2.8298 | 2.5546 | -0.2753 |
| adam-124m-gelu-symmetric-seed42 | 2.8558 | 2.5361 | -0.3197 |
| ecd-124m-gelu-bQonly-seed83 | 3.1279 | 2.9126 | -0.2153 |
| ecd-124m-gelu-bQonly-seed42 | 3.1365 | 3.0029 | -0.1336 |
| ecd-124m-gelu-symmetric-seed83 | 3.1407 | 2.7102 | -0.4305 |
| ecd-124m-gelu-bQbV-meanV00-seed83 | 3.1422 | 2.8167 | -0.3255 |
| ecd-124m-gelu-bQbV-meanV05-seed83 | 3.1479 | 3.3621 | +0.2142 |
| ecd-124m-gelu-bQbV-meanV05-seed789 | 3.1515 | 2.9556 | -0.1959 |
| ecd-124m-gelu-bQbV-meanV00-seed789 | 3.1553 | 2.8196 | -0.3357 |
| ecd-124m-gelu-bQonly-seed789 | 3.1556 | 2.8225 | -0.3332 |
| ecd-124m-gelu-bQbV-seed42 | 3.1607 | 3.0947 | -0.0660 |
| ecd-124m-gelu-symmetric-seed789 | 3.1761 | 3.1109 | -0.0652 |
| ecd-124m-gelu-symmetric-seed42 | 3.1852 | 2.8367 | -0.3485 |
| soap-124m-gelu-symmetric-seed42 | 3.3130 | 3.7170 | +0.4040 |
| soap-124m-gelu-bQbV-seed42 | 3.3144 | 3.7659 | +0.4515 |
| sgdm-124m-gelu-symmetric-seed42 | 3.5524 | 3.6152 | +0.0628 |
| sgdm-124m-gelu-bQbV-seed42 | 3.5634 | 3.6749 | +0.1115 |

## Grouped by Optimizer

### ADAM

| Model | Training Val Loss | Logic Puzzle Loss |
|-------|------------------|-------------------|
| adam-124m-gelu-bQbV-seed42 | 2.8298 | 2.5546 |
| adam-124m-gelu-symmetric-seed42 | 2.8558 | 2.5361 |

### ECD

| Model | Training Val Loss | Logic Puzzle Loss |
|-------|------------------|-------------------|
| ecd-124m-gelu-bQonly-seed83 | 3.1279 | 2.9126 |
| ecd-124m-gelu-bQonly-seed42 | 3.1365 | 3.0029 |
| ecd-124m-gelu-symmetric-seed83 | 3.1407 | 2.7102 |
| ecd-124m-gelu-bQbV-meanV00-seed83 | 3.1422 | 2.8167 |
| ecd-124m-gelu-bQbV-meanV05-seed83 | 3.1479 | 3.3621 |
| ecd-124m-gelu-bQbV-meanV05-seed789 | 3.1515 | 2.9556 |
| ecd-124m-gelu-bQbV-meanV00-seed789 | 3.1553 | 2.8196 |
| ecd-124m-gelu-bQonly-seed789 | 3.1556 | 2.8225 |
| ecd-124m-gelu-bQbV-seed42 | 3.1607 | 3.0947 |
| ecd-124m-gelu-symmetric-seed789 | 3.1761 | 3.1109 |
| ecd-124m-gelu-symmetric-seed42 | 3.1852 | 2.8367 |

### SGDM

| Model | Training Val Loss | Logic Puzzle Loss |
|-------|------------------|-------------------|
| sgdm-124m-gelu-symmetric-seed42 | 3.5524 | 3.6152 |
| sgdm-124m-gelu-bQbV-seed42 | 3.5634 | 3.6749 |

### SOAP

| Model | Training Val Loss | Logic Puzzle Loss |
|-------|------------------|-------------------|
| soap-124m-gelu-symmetric-seed42 | 3.3130 | 3.7170 |
| soap-124m-gelu-bQbV-seed42 | 3.3144 | 3.7659 |

## Grouped by Configuration

### Symmetric (no bQ, no bV)

| Model | Training Val Loss | Logic Puzzle Loss |
|-------|------------------|-------------------|
| adam-124m-gelu-symmetric-seed42 | 2.8558 | 2.5361 |
| ecd-124m-gelu-symmetric-seed83 | 3.1407 | 2.7102 |
| ecd-124m-gelu-symmetric-seed789 | 3.1761 | 3.1109 |
| ecd-124m-gelu-symmetric-seed42 | 3.1852 | 2.8367 |
| soap-124m-gelu-symmetric-seed42 | 3.3130 | 3.7170 |
| sgdm-124m-gelu-symmetric-seed42 | 3.5524 | 3.6152 |

### bQ only (no bV)

| Model | Training Val Loss | Logic Puzzle Loss |
|-------|------------------|-------------------|
| ecd-124m-gelu-bQonly-seed83 | 3.1279 | 2.9126 |
| ecd-124m-gelu-bQonly-seed42 | 3.1365 | 3.0029 |
| ecd-124m-gelu-bQonly-seed789 | 3.1556 | 2.8225 |

### bQ+bV (mean_V=0.0, seed 42)

| Model | Training Val Loss | Logic Puzzle Loss |
|-------|------------------|-------------------|
| adam-124m-gelu-bQbV-seed42 | 2.8298 | 2.5546 |
| ecd-124m-gelu-bQbV-seed42 | 3.1607 | 3.0947 |
| soap-124m-gelu-bQbV-seed42 | 3.3144 | 3.7659 |
| sgdm-124m-gelu-bQbV-seed42 | 3.5634 | 3.6749 |

### bQ+bV (mean_V=0.0)

| Model | Training Val Loss | Logic Puzzle Loss |
|-------|------------------|-------------------|
| ecd-124m-gelu-bQbV-meanV00-seed83 | 3.1422 | 2.8167 |
| ecd-124m-gelu-bQbV-meanV00-seed789 | 3.1553 | 2.8196 |

### bQ+bV (mean_V=0.5)

| Model | Training Val Loss | Logic Puzzle Loss |
|-------|------------------|-------------------|
| ecd-124m-gelu-bQbV-meanV05-seed83 | 3.1479 | 3.3621 |
| ecd-124m-gelu-bQbV-meanV05-seed789 | 3.1515 | 2.9556 |

## Logic Puzzle Task Breakdown

Tasks evaluated (14 total):
- pattern_numeric (4 tasks)
- pattern_alpha (2 tasks)
- retrieval_near (2 tasks)
- retrieval_far (2 tasks)
- simple_inference (2 tasks)
- negation (3 tasks)
- syntax (2 tasks)
- copy (2 tasks)

## Notes

- **Training Val Loss**: Best validation loss achieved during training (cross-entropy on held-out data)
- **Logic Puzzle Loss**: Average loss across 14 logic puzzle tasks
- **Δ**: Difference between logic puzzle loss and training val loss (positive = worse on logic puzzles)
- All models use GELU MLP (this is an ablation comparing to PReLU models)
- Models trained on 500M tokens of FineWebEdu
