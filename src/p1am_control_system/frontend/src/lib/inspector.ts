export type InspectorState =
  | { type: "none" }
  | { type: "tag"; tagId: number }
  | { type: "custom_tag"; tagName: string }
  | { type: "pid"; index: number }
  | { type: "routing" }
  | { type: "alicat"; deviceId: string }
  | { type: "settings" }
  | { type: "export" };

