#!/usr/bin/env python3
"""
Comprehensive Dashboard Audit Script
Analyzes the SUHI dashboard for:
1. Real data usage vs mock/dummy data
2. Naming consistency across tabs
3. Data integrity and completeness
4. Cross-tab data validation
"""

import re
import os
import json
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

class DashboardAuditor:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.data_sources = {}
        self.naming_patterns = defaultdict(set)
        self.city_names = set()
        self.variable_names = set()
        
    def log_issue(self, severity: str, category: str, message: str, location: str = ""):
        """Log an audit issue"""
        self.issues.append({
            'severity': severity,
            'category': category,
            'message': message,
            'location': location
        })
    
    def audit_html_structure(self, html_content: str):
        """Audit HTML structure for naming consistency"""
        print("🔍 Auditing HTML Structure...")
        
        # Extract tab names and IDs
        tab_pattern = r'data-section="([^"]+)"'
        tabs = re.findall(tab_pattern, html_content)
        print(f"   Found tabs: {tabs}")
        
        # Extract element IDs
        id_pattern = r'id="([^"]+)"'
        element_ids = re.findall(id_pattern, html_content)
        
        # Check for consistent naming patterns
        city_related_ids = [eid for eid in element_ids if 'city' in eid.lower()]
        suhi_related_ids = [eid for eid in element_ids if 'suhi' in eid.lower()]
        
        print(f"   City-related IDs: {len(city_related_ids)}")
        print(f"   SUHI-related IDs: {len(suhi_related_ids)}")
        
        # Check for inconsistent naming patterns
        inconsistent_patterns = []
        for eid in element_ids:
            if 'temp' in eid.lower() and 'temperature' in eid.lower():
                inconsistent_patterns.append(f"Mixed temp/temperature: {eid}")
            if 'suhi' in eid.lower() and 'uhi' in eid.lower():
                inconsistent_patterns.append(f"Mixed SUHI/UHI: {eid}")
        
        if inconsistent_patterns:
            for pattern in inconsistent_patterns[:5]:  # Show first 5
                self.log_issue("WARNING", "Naming", pattern, "HTML")
    
    def audit_javascript_data(self, js_content: str):
        """Audit JavaScript data for real vs mock data"""
        print("🔍 Auditing JavaScript Data Sources...")
        
        # Extract suhiData object
        data_match = re.search(r'const suhiData = ({.*?});', js_content, re.DOTALL)
        if not data_match:
            self.log_issue("ERROR", "Data", "suhiData object not found", "JavaScript")
            return
        
        # Check for mock data patterns
        mock_patterns = [
            r'mock\w*',
            r'fake\w*',
            r'dummy\w*',
            r'test\w*data',
            r'sample\w*data',
            r'lorem\w*',
            r'placeholder'
        ]
        
        mock_found = []
        for pattern in mock_patterns:
            matches = re.findall(pattern, js_content, re.IGNORECASE)
            if matches:
                mock_found.extend(matches)
        
        if mock_found:
            for mock in set(mock_found):
                self.log_issue("ERROR", "Data", f"Mock data detected: {mock}", "JavaScript")
        
        # Extract city names from different sections
        self.extract_city_names(js_content)
        
        # Check data completeness
        self.check_data_completeness(js_content)
    
    def extract_city_names(self, js_content: str):
        """Extract city names from all sections of JavaScript"""
        print("🔍 Extracting City Names from All Sections...")
        
        # From cities array
        cities_match = re.search(r'"cities":\s*\[(.*?)\]', js_content, re.DOTALL)
        if cities_match:
            city_objects = re.findall(r'"name":\s*"([^"]+)"', cities_match.group(1))
            self.naming_patterns['cities_array'].update(city_objects)
            print(f"   Cities array: {len(city_objects)} cities")
        
        # From timeSeriesData
        timeseries_match = re.search(r'"timeSeriesData":\s*{(.*?)\n  },', js_content, re.DOTALL)
        if timeseries_match:
            timeseries_cities = re.findall(r'\n    "([^"]+)":\s*{', timeseries_match.group(1))
            self.naming_patterns['timeseries_data'].update(timeseries_cities)
            print(f"   TimeSeriesData: {len(timeseries_cities)} cities")
        
        # From yearOverYearChanges
        changes_match = re.search(r'"yearOverYearChanges":\s*\[(.*?)\]', js_content, re.DOTALL)
        if changes_match:
            change_cities = re.findall(r'"city":\s*"([^"]+)"', changes_match.group(1))
            self.naming_patterns['year_changes'].update(change_cities)
            print(f"   Year-over-year changes: {len(set(change_cities))} unique cities")
        
        # Consolidate all city names
        all_cities = set()
        for source, cities in self.naming_patterns.items():
            all_cities.update(cities)
        self.city_names = all_cities
        print(f"   Total unique cities found: {len(all_cities)}")
    
    def check_data_completeness(self, js_content: str):
        """Check data completeness and consistency"""
        print("🔍 Checking Data Completeness...")
        
        # Check if all data sources have the same cities
        sources = ['cities_array', 'timeseries_data', 'year_changes']
        city_counts = {}
        
        for source in sources:
            if source in self.naming_patterns:
                city_counts[source] = len(self.naming_patterns[source])
        
        # Check for consistency
        expected_count = 14  # We know we should have 14 cities
        for source, count in city_counts.items():
            if count != expected_count:
                self.log_issue("WARNING", "Data", 
                             f"{source} has {count} cities, expected {expected_count}", 
                             "JavaScript")
        
        # Check for missing cities across sources
        if len(self.naming_patterns) >= 2:
            base_cities = self.naming_patterns['cities_array']
            for source, cities in self.naming_patterns.items():
                if source != 'cities_array':
                    missing = base_cities - cities
                    extra = cities - base_cities
                    
                    if missing:
                        self.log_issue("ERROR", "Data", 
                                     f"{source} missing cities: {missing}", 
                                     "JavaScript")
                    if extra:
                        self.log_issue("WARNING", "Data", 
                                     f"{source} has extra cities: {extra}", 
                                     "JavaScript")
    
    def audit_chart_functions(self, js_content: str):
        """Audit chart functions for data usage consistency"""
        print("🔍 Auditing Chart Functions...")
        
        # Find all chart loading functions
        chart_functions = re.findall(r'function (load\w*Chart)\(\)', js_content)
        print(f"   Found {len(chart_functions)} chart functions")
        
        for func_name in chart_functions:
            func_match = re.search(f'function {func_name}\\(\\)(.*?)\\n}}', js_content, re.DOTALL)
            if func_match:
                func_content = func_match.group(1)
                
                # Check data source usage
                if 'suhiData.cities' in func_content:
                    data_source = "real_cities_data"
                elif 'suhiData.timeSeriesData' in func_content:
                    data_source = "real_timeseries_data"
                elif any(mock in func_content.lower() for mock in ['mock', 'fake', 'dummy']):
                    data_source = "mock_data"
                    self.log_issue("ERROR", "Chart", f"{func_name} uses mock data", "JavaScript")
                else:
                    data_source = "unknown"
                    self.log_issue("WARNING", "Chart", f"{func_name} has unclear data source", "JavaScript")
                
                print(f"     {func_name}: {data_source}")
    
    def audit_variable_consistency(self, js_content: str):
        """Audit variable naming consistency"""
        print("🔍 Auditing Variable Naming Consistency...")
        
        # Temperature/SUHI related variables
        temp_vars = re.findall(r'\b(\w*[Tt]emp\w*)\b', js_content)
        suhi_vars = re.findall(r'\b(\w*[Ss][Uu][Hh][Ii]\w*)\b', js_content)
        
        # Check for inconsistent temperature naming
        temp_patterns = set()
        for var in temp_vars:
            if 'temp' in var.lower():
                if 'day' in var.lower():
                    temp_patterns.add('day_temp_pattern')
                elif 'night' in var.lower():
                    temp_patterns.add('night_temp_pattern')
        
        # Look for inconsistent SUHI naming
        suhi_naming_issues = []
        for var in suhi_vars:
            if 'day' in var.lower() and 'Day' in var:
                suhi_naming_issues.append(f"Mixed case in {var}")
            if 'suhi' in var.lower() and 'SUHI' in var:
                suhi_naming_issues.append(f"Mixed SUHI case in {var}")
        
        for issue in suhi_naming_issues[:5]:  # Show first 5
            self.log_issue("WARNING", "Naming", issue, "JavaScript")
    
    def audit_csv_data_sources(self):
        """Audit actual CSV data files for authenticity"""
        print("🔍 Auditing CSV Data Sources...")
        
        csv_dir = "scientific_suhi_analysis/data/"
        if not os.path.exists(csv_dir):
            self.log_issue("ERROR", "Data", f"CSV directory not found: {csv_dir}", "FileSystem")
            return
        
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        print(f"   Found {len(csv_files)} CSV files")
        
        if len(csv_files) < 10:
            self.log_issue("WARNING", "Data", f"Expected ~10 CSV files, found {len(csv_files)}", "FileSystem")
        
        # Check file naming pattern
        expected_pattern = r'suhi_data_period_\d{4}_\d{8}_\d{6}\.csv'
        for csv_file in csv_files:
            if not re.match(expected_pattern, csv_file):
                self.log_issue("WARNING", "Naming", f"Unexpected CSV filename: {csv_file}", "FileSystem")
    
    def generate_report(self):
        """Generate comprehensive audit report"""
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE DASHBOARD AUDIT REPORT")
        print("="*60)
        
        # Summary statistics
        error_count = len([i for i in self.issues if i['severity'] == 'ERROR'])
        warning_count = len([i for i in self.issues if i['severity'] == 'WARNING'])
        
        print(f"\n📊 AUDIT SUMMARY")
        print(f"   Total Issues Found: {len(self.issues)}")
        print(f"   Errors (Critical): {error_count}")
        print(f"   Warnings (Review): {warning_count}")
        
        # Data integrity summary
        print(f"\n🎯 DATA INTEGRITY")
        print(f"   Unique Cities Found: {len(self.city_names)}")
        print(f"   Data Sources Checked: {len(self.naming_patterns)}")
        
        if self.city_names:
            print(f"   Cities: {', '.join(sorted(list(self.city_names))[:5])}{'...' if len(self.city_names) > 5 else ''}")
        
        # Issues by category
        if self.issues:
            print(f"\n🔍 ISSUES BY CATEGORY")
            category_counts = Counter(issue['category'] for issue in self.issues)
            for category, count in category_counts.most_common():
                print(f"   {category}: {count} issues")
        
        # Detailed issues
        if error_count > 0:
            print(f"\n❌ CRITICAL ERRORS")
            for issue in self.issues:
                if issue['severity'] == 'ERROR':
                    location = f" ({issue['location']})" if issue['location'] else ""
                    print(f"   • {issue['category']}: {issue['message']}{location}")
        
        if warning_count > 0:
            print(f"\n⚠️  WARNINGS")
            for issue in self.issues:
                if issue['severity'] == 'WARNING':
                    location = f" ({issue['location']})" if issue['location'] else ""
                    print(f"   • {issue['category']}: {issue['message']}{location}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT")
        if error_count == 0:
            print("   ✅ No critical errors found")
        else:
            print(f"   ❌ {error_count} critical errors require immediate attention")
        
        if warning_count == 0:
            print("   ✅ No warnings")
        else:
            print(f"   ⚠️  {warning_count} warnings should be reviewed")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        if error_count == 0 and warning_count == 0:
            print("   🎉 Dashboard audit passed! No issues detected.")
        else:
            if error_count > 0:
                print("   1. Address all critical errors immediately")
            if warning_count > 0:
                print("   2. Review and resolve warnings for better consistency")
            print("   3. Re-run audit after fixes")

def main():
    """Main audit function"""
    auditor = DashboardAuditor()
    
    print("🔍 STARTING COMPREHENSIVE DASHBOARD AUDIT")
    print("="*50)
    
    # Read HTML file
    html_file = "index.html"
    if os.path.exists(html_file):
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        auditor.audit_html_structure(html_content)
    else:
        auditor.log_issue("ERROR", "File", f"HTML file not found: {html_file}", "FileSystem")
    
    # Read JavaScript file
    js_file = "enhanced-suhi-dashboard.js"
    if os.path.exists(js_file):
        with open(js_file, 'r', encoding='utf-8') as f:
            js_content = f.read()
        auditor.audit_javascript_data(js_content)
        auditor.audit_chart_functions(js_content)
        auditor.audit_variable_consistency(js_content)
    else:
        auditor.log_issue("ERROR", "File", f"JavaScript file not found: {js_file}", "FileSystem")
    
    # Audit CSV data sources
    auditor.audit_csv_data_sources()
    
    # Generate comprehensive report
    auditor.generate_report()

if __name__ == "__main__":
    main()
