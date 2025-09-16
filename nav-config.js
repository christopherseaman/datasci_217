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
                { file: "01/index", label: "1: Command Line + Python" },
                { file: "02/index", label: "2: Python + Git" },
                { file: "03/index", label: "3: Data Structures" },
                { file: "04/index", label: "4: NumPy" },
                { file: "05/index", label: "5: Pandas" }
            ]
        },
        {
            title: "Extended Track",
            links: [
                { file: "06/index", label: "6: Data Loading" },
                { file: "07/index", label: "7: Data Cleaning" },
                { file: "08/index", label: "8: Data Wrangling" },
                { file: "09/index", label: "9: Visualization" },
                { file: "10/index", label: "10: Aggregation" },
                { file: "11/index", label: "11: Time Series" }
            ]
        },
        {
            title: "Resources",
            links: [
                { file: "https://github.com/christopherseaman/datasci_217", label: "GitHub Repository", external: true },
                { file: "https://wesmckinney.com/book/", label: "McKinney Book", external: true }
            ]
        }
    ],
    pageTitles: {
        'README': 'Course Overview',
        '01/index': '1: Command Line + Python',
        '02/index': '2: Python + Git',
        '03/index': '3: Data Structures',
        '04/index': '4: NumPy',
        '05/index': '5: Pandas',
        '06/index': '6: Data Loading',
        '07/index': '7: Data Cleaning',
        '08/index': '8: Data Wrangling',
        '09/index': '9: Visualization',
        '10/index': '10: Aggregation',
        '11/index': '11: Time Series'
    }
};
