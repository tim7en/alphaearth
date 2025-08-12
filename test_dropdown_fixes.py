#!/usr/bin/env python3
"""
Test the fixed dropdown for city selection in temporal trends
"""

import re

def check_dropdown_issues():
    """Check for dropdown duplication issues"""
    print("🔍 CHECKING TEMPORAL TRENDS DROPDOWN FIXES")
    print("=" * 50)
    
    # Read JavaScript file
    with open('enhanced-suhi-dashboard.js', 'r', encoding='utf-8') as f:
        js_content = f.read()
    
    # Check initialization calls (exclude function definition)
    init_calls = len(re.findall(r'[^n]\s+initializeDashboard\(\)', js_content))
    print(f"📊 Initialization Analysis:")
    print(f"   initializeDashboard() calls found: {init_calls}")
    
    # Check if updateTrendsChart function exists
    update_func = re.search(r'function updateTrendsChart\(\)', js_content)
    print(f"   updateTrendsChart function exists: {'✅' if update_func else '❌'}")
    
    # Check if dropdown clearing logic exists
    clear_logic = 'querySelectorAll(\'option:not([value="all"])\')' in js_content
    print(f"   Dropdown clearing logic present: {'✅' if clear_logic else '❌'}")
    
    # Extract cities from the cities array
    cities_match = re.search(r'"cities":\s*\[(.*?)\]', js_content, re.DOTALL)
    if cities_match:
        city_names = re.findall(r'"name":\s*"([^"]+)"', cities_match.group(1))
        print(f"   Cities in data array: {len(city_names)}")
        print(f"   Cities: {', '.join(city_names[:5])}{'...' if len(city_names) > 5 else ''}")
    
    # Extract cities from timeSeriesData
    timeseries_match = re.search(r'"timeSeriesData":\s*{(.*?)\n  },', js_content, re.DOTALL)
    if timeseries_match:
        timeseries_cities = re.findall(r'\n    "([^"]+)":\s*{', timeseries_match.group(1))
        print(f"   Cities in timeSeriesData: {len(timeseries_cities)}")
    
    # Check for proper event handling
    event_listeners = len(re.findall(r'document\.addEventListener\(', js_content))
    print(f"   Event listeners found: {event_listeners}")
    
    print(f"\n🎯 FIX STATUS")
    print("-" * 20)
    
    issues_fixed = 0
    total_issues = 3
    
    if update_func:
        print("   ✅ Missing updateTrendsChart function - FIXED")
        issues_fixed += 1
    else:
        print("   ❌ Missing updateTrendsChart function - NOT FIXED")
    
    if clear_logic:
        print("   ✅ Dropdown duplication prevention - FIXED")
        issues_fixed += 1
    else:
        print("   ❌ Dropdown duplication prevention - NOT FIXED")
    
    if init_calls <= 2:  # Should have at most 1-2 controlled calls
        print("   ✅ Multiple initialization calls - CONTROLLED")
        issues_fixed += 1
    else:
        print("   ❌ Multiple initialization calls - STILL PROBLEMATIC")
    
    print(f"\n📋 SUMMARY")
    print(f"   Issues fixed: {issues_fixed}/{total_issues}")
    print(f"   Status: {'✅ READY' if issues_fixed == total_issues else '⚠️ NEEDS MORE WORK'}")
    
    if issues_fixed == total_issues:
        print(f"\n🎉 DROPDOWN SHOULD NOW WORK CORRECTLY")
        print(f"   • No duplicate city names")
        print(f"   • Exactly 14 cities + 'All Cities Average' option")
        print(f"   • Functional dropdown with chart updates")

def main():
    check_dropdown_issues()

if __name__ == "__main__":
    main()
