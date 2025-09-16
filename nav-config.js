// Navigation configuration - edit this file to update navigation across all pages
const navConfig = {
    logo: "DataSci-217",
    sections: [
        {
            title: "Course Overview",
            links: [
                { file: "README", label: "Course Overview" }
            ]
        },
        {
            title: "Foundational Track",
            links: [
                { file: "01/", label: "1: Command Line + Python" },
                { file: "02/", label: "2: Python + Git" },
                { file: "03/", label: "3: Data Structures" },
                { file: "04/", label: "4: NumPy" },
                { file: "05/", label: "5: Pandas" }
            ]
        },
        {
            title: "Extended Track",
            links: [
                { file: "06/", label: "6: Data Loading" },
                { file: "07/", label: "7: Data Cleaning" },
                { file: "08/", label: "8: Data Wrangling" },
                { file: "09/", label: "9: Visualization" },
                { file: "10/", label: "10: Aggregation" },
                { file: "11/", label: "11: Time Series" }
            ]
        },
        {
            title: "Resources",
            links: [
                { file: "https://github.com/christopherseaman/datasci_217", label: "GitHub Repository", external: true },
                { file: "https://wesmckinney.com/book/", label: "McKinney Book", external: true },
                { file: "https://missing.csail.mit.edu/", label: "The Missing Semester", external: true },
                { file: "http://linuxcommand.org/tlcl.php", label: "The Linux Command Line", external: true }
            ]
        }
    ],
    pageTitles: {
        'README': 'Course Overview',
        '01/': '1: Command Line + Python',
        '02//': '2: Python + Git',
        '03/': '3: Data Structures',
        '04/': '4: NumPy',
        '05/': '5: Pandas',
        '06/': '6: Data Loading',
        '07/': '7: Data Cleaning',
        '08/': '8: Data Wrangling',
        '09/': '9: Visualization',
        '10/': '10: Aggregation',
        '11/': '11: Time Series'
    }
};
