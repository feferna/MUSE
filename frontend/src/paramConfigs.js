const PARAM_CONFIGS = {
  duration: {
    label: 'Animation Duration',
    type: 'slider',
    min: 100,
    max: 1000,
    step: 50,
    unit: 'ms',
    default: 350,
  },
  borderRadius: {
    label: 'Border Radius',
    type: 'slider',
    min: 0,
    max: 50,
    step: 1,
    unit: 'px',
    default: 0,
  },
  thumbRatio: {
    label: 'Thumb Size Ratio',
    type: 'slider',
    min: 0.3,
    max: 0.95,
    step: 0.05,
    unit: '',
    default: 0.8,
  },
  thumbShape: {
    label: 'Thumb Shape',
    type: 'cards',
    options: [
      {value: 'circle', label: 'Circle', desc: 'Perfect round'},
      {value: 'square', label: 'Square', desc: 'Sharp corners'},
      {value: 'diamond', label: 'Diamond', desc: 'Rotated square'},
      {value: 'hexagon', label: 'Hexagon', desc: 'Six-sided'},
      {value: 'star', label: 'Star', desc: 'Five-pointed'},
      {value: 'triangle', label: 'Triangle', desc: 'Equilateral'},
      {value: 'teardrop', label: 'Teardrop', desc: 'Pointed drop'},
      {value: 'bean', label: 'Bean', desc: 'Organic blob'},
    ],
    default: 'circle',
  },
  easing: {
    label: 'Easing Curve',
    type: 'dropdown',
    options: [
      {value: 'linear', label: 'Linear (Constant speed)'},
      {value: 'ease', label: 'Ease (Default Smooth)'},
      {value: 'ease-in', label: 'Ease In (Slow Start)'},
      {value: 'ease-out', label: 'Ease Out (Slow End)'},
      {value: 'ease-in-out', label: 'Ease In-Out (Smooth Both Ends)'},
    ],
    default: 'ease-out',
  },
  visualStyle: {
    label: 'Visual Style',
    type: 'cards',
    options: [
      {
        value: 'flat-solid',
        label: 'Flat Solid',
        desc: 'Clean minimal',
        style: {visual: 'flat', track: 'solid'},
      },
      {
        value: 'shadow-gradient',
        label: 'Shadow Gradient',
        desc: 'Depth+gradient',
        style: {visual: 'shadow', track: 'gradient'},
      },
      {
        value: 'glow-glass',
        label: 'Glow Glass',
        desc: 'Luminous glass',
        style: {visual: 'glow', track: 'glass'},
      },
      {
        value: 'outline-outlined',
        label: 'Outline Style',
        desc: 'Bordered',
        style: {visual: 'outline', track: 'outlined'},
      },
      {
        value: 'elevated-textured',
        label: 'Elevated Textured',
        desc: 'Floating feel',
        style: {visual: 'elevated', track: 'textured'},
      },
    ],
    default: 'shadow-gradient',
  },
  colorScheme: {
    label: 'Color Scheme',
    type: 'colorGrid',
    options: [
      {
        value: 'modern',
        label: 'Modern Blue',
        colors: ['#3b82f6', '#e5e7eb', '#ffffff'],
      },
      {
        value: 'nature',
        label: 'Nature Green',
        colors: ['#10b981', '#d1fae5', '#ffffff'],
      },
      {
        value: 'sunset',
        label: 'Sunset Orange',
        colors: ['#f97316', '#fed7aa', '#ffffff'],
      },
      {
        value: 'ocean',
        label: 'Ocean Teal',
        colors: ['#0891b2', '#cffafe', '#ffffff'],
      },
      {
        value: 'dark',
        label: 'Dark Mode',
        colors: ['#6366f1', '#374151', '#1f2937'],
      },
      {
        value: 'neon',
        label: 'Neon Purple',
        colors: ['#a855f7', '#1a1a1a', '#e879f9'],
      },
      {
        value: 'enterprise',
        label: 'Enterprise',
        colors: ['#6b7280', '#f3f4f6', '#ffffff'],
      },
    ],
    default: 'modern',
  },
};

export default PARAM_CONFIGS;
