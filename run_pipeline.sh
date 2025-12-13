#!/bin/bash

# ============================================================================
# NFL Big Data Bowl 2026 - Pipeline Runner (Bash Script)
# ============================================================================
# 
# Simple bash script alternative to run_it.py
# No Airflow required - just runs scripts sequentially
#
# USAGE:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo "================================================================================"
    echo -e "${BLUE}$1${NC}"
    echo "================================================================================"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

check_file() {
    if [ -f "$1" ]; then
        print_success "Found: $1"
        return 0
    else
        print_error "Missing: $1"
        return 1
    fi
}

# ============================================================================
# Start Pipeline
# ============================================================================

START_TIME=$(date +%s)

print_header "NFL BIG DATA BOWL 2026 - DATA PIPELINE"
echo "Start time: $(date)"
echo ""

# ============================================================================
# Step 0: Check Prerequisites
# ============================================================================

print_header "STEP 0: CHECKING PREREQUISITES"

MISSING_FILES=0

echo "Checking required data files..."
check_file "data/train/2023_input_all.parquet" || ((MISSING_FILES++))
check_file "data/supplementary_data.csv" || ((MISSING_FILES++))
check_file "data/sumer_bdb/sumer_coverages_player_play.parquet" || ((MISSING_FILES++))
check_file "data/sumer_bdb/sumer_coverages_frame.parquet" || ((MISSING_FILES++))

echo ""
echo "Checking required scripts..."
check_file "dataframe_a_v2.py" || ((MISSING_FILES++))
check_file "dataframe_b_v3.py" || ((MISSING_FILES++))
check_file "dataframe_c_v3.py" || ((MISSING_FILES++))
check_file "dataframe_d.py" || ((MISSING_FILES++))

if [ $MISSING_FILES -gt 0 ]; then
    print_error "Missing $MISSING_FILES required files!"
    exit 1
fi

print_success "All prerequisites satisfied!"

# ============================================================================
# Step 1: Create Output Directories
# ============================================================================

print_header "STEP 1: CREATING OUTPUT DIRECTORIES"

mkdir -p outputs/dataframe_a
mkdir -p outputs/dataframe_b
mkdir -p outputs/dataframe_c
mkdir -p outputs/dataframe_d

print_success "Output directories created"

# ============================================================================
# Step 2: Run Dataframe A
# ============================================================================

print_header "STEP 2: RUNNING DATAFRAME A (v2) - NODE-LEVEL FEATURES"

python dataframe_a_v2.py

if [ $? -eq 0 ]; then
    print_success "Dataframe A completed successfully"
    
    # Check output
    if [ -f "outputs/dataframe_a/v2.parquet" ]; then
        SIZE=$(ls -lh "outputs/dataframe_a/v2.parquet" | awk '{print $5}')
        print_success "Output file created: $SIZE"
    else
        print_error "Expected output file not found!"
        exit 1
    fi
else
    print_error "Dataframe A failed!"
    exit 1
fi

# ============================================================================
# Step 3: Run Dataframe B
# ============================================================================

print_header "STEP 3: RUNNING DATAFRAME B (v3) - PLAY-LEVEL + BALL TRAJECTORY"

python dataframe_b_v3.py

if [ $? -eq 0 ]; then
    print_success "Dataframe B completed successfully"
    
    # Check for output (pilot or full)
    if [ -f "outputs/dataframe_b/v3_pilot_3games.parquet" ]; then
        SIZE=$(ls -lh "outputs/dataframe_b/v3_pilot_3games.parquet" | awk '{print $5}')
        print_warning "Pilot mode detected: $SIZE"
    elif [ -f "outputs/dataframe_b/v3.parquet" ]; then
        SIZE=$(ls -lh "outputs/dataframe_b/v3.parquet" | awk '{print $5}')
        print_success "Full output created: $SIZE"
    else
        print_error "Expected output file not found!"
        exit 1
    fi
else
    print_error "Dataframe B failed!"
    exit 1
fi

# ============================================================================
# Step 4: Run Dataframe D (parallel-safe)
# ============================================================================

print_header "STEP 4: RUNNING DATAFRAME D (v2) - FRAME-LEVEL PLAYER COUNTS"

python dataframe_d.py

if [ $? -eq 0 ]; then
    print_success "Dataframe D completed successfully"
    
    if [ -f "outputs/dataframe_d/v1.parquet" ]; then
        SIZE=$(ls -lh "outputs/dataframe_d/v1.parquet" | awk '{print $5}')
        print_success "Output file created: $SIZE"
    else
        print_error "Expected output file not found!"
        exit 1
    fi
else
    print_error "Dataframe D failed!"
    exit 1
fi

# ============================================================================
# Step 5: Run Dataframe C
# ============================================================================

print_header "STEP 5: RUNNING DATAFRAME C (v3) - EDGE-LEVEL + BALL TRAJECTORY"

python dataframe_c_v3.py

if [ $? -eq 0 ]; then
    print_success "Dataframe C completed successfully"
    
    # Check for output (pilot or full)
    if [ -f "outputs/dataframe_c/v3_pilot_3games.parquet" ]; then
        SIZE=$(ls -lh "outputs/dataframe_c/v3_pilot_3games.parquet" | awk '{print $5}')
        print_warning "Pilot mode detected: $SIZE"
    elif [ -f "outputs/dataframe_c/v3.parquet" ]; then
        SIZE=$(ls -lh "outputs/dataframe_c/v3.parquet" | awk '{print $5}')
        print_success "Full output created: $SIZE"
    else
        print_error "Expected output file not found!"
        exit 1
    fi
else
    print_error "Dataframe C failed!"
    exit 1
fi

# ============================================================================
# Step 6: Generate Summary Report
# ============================================================================

print_header "STEP 6: PIPELINE SUMMARY REPORT"

echo "Output Files:"
echo "--------------------------------------------------------------------------------"
printf "%-40s %-15s %-15s\n" "File" "Exists" "Size"
echo "--------------------------------------------------------------------------------"

check_output() {
    local name=$1
    local path=$2
    
    if [ -f "$path" ]; then
        size=$(ls -lh "$path" | awk '{print $5}')
        printf "%-40s ${GREEN}%-15s${NC} %-15s\n" "$name" "✓" "$size"
    else
        printf "%-40s ${RED}%-15s${NC} %-15s\n" "$name" "✗" "-"
    fi
}

check_output "Dataframe A (v2)" "outputs/dataframe_a/v2.parquet"
check_output "Dataframe B (v3)" "outputs/dataframe_b/v3.parquet"
check_output "Dataframe B (v3 pilot)" "outputs/dataframe_b/v3_pilot_3games.parquet"
check_output "Dataframe C (v3)" "outputs/dataframe_c/v3.parquet"
check_output "Dataframe C (v3 pilot)" "outputs/dataframe_c/v3_pilot_3games.parquet"
check_output "Dataframe D (v1)" "outputs/dataframe_d/v1.parquet"

# ============================================================================
# Complete
# ============================================================================

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

print_header "PIPELINE COMPLETE!"

echo "Summary:"
echo "  Start time: $(date -d @$START_TIME)"
echo "  End time:   $(date -d @$END_TIME)"
echo "  Duration:   ${MINUTES}m ${SECONDS}s"
echo ""
print_success "All dataframes generated successfully!"
echo ""
echo "Next steps:"
echo "  1. Verify outputs in outputs/ directory"
echo "  2. Check individual parquet files"
echo "  3. Begin attention mechanism development"
echo ""
echo "================================================================================"
