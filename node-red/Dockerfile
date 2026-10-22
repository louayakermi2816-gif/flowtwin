# Node-RED with the FlowTwin supervision flow and FlowFuse Dashboard 2.0 built in:
# the dashboard is at http://localhost:1880/dashboard, the editor at http://localhost:1880.
FROM nodered/node-red:4.1.15-22
RUN npm install --no-audit --no-fund @flowfuse/node-red-dashboard@1.31.0
COPY --chown=node-red:node-red flows.json /data/flows.json
